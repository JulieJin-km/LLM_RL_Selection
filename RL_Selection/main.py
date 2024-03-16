import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import argparse
import torch
import numpy as np
import random
import logging
from torch.utils.data import SequentialSampler,DataLoader,RandomSampler
from transformers import AutoTokenizer,AutoConfig
from tqdm import tqdm,trange
from torch.distributions.binomial import Binomial

import copy
import json
import pytorch_lightning as pl
import transformers

from dataset import RLDataset, SquadDataset
from agent import BertForSequenceClassification,RLAgent_BERT
from reward import QA_Performance_Enviroment,BertForQuestionAnswering,SummarizationModule,GEN_QA_Performance_Enviroment,LLMEnviroment,OptLLMEnviroment,StrictEnvironment
from Scorers import FLANScorer,OPTScorer,GPT3Scorer

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_error()



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_argument():
    parser = argparse.ArgumentParser()
    #这个parser需要手动开关，只有在训练生成式数据+QA作为reward时会使用，如果开的话，要把seed注释掉，不然会重复
    #parser = pl.Trainer.add_argparse_args(parser)
    #parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--enviroment_type",default='ext',type=str,help="extractive mrc or genrative mrc or LLM")
    parser.add_argument(
        "--rl_output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--do_rl_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_rl_estimation", action="store_true",
                        help="Whether to question value estimation for the training set")
    parser.add_argument(
        "--rl_cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )


    # For Dataset
    parser.add_argument(
        "--QAG_file",
        default=None,
        type=str,
        help="The input training file.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="The name of dataset. ",
    )
    parser.add_argument(
        "--rl_tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )

    parser.add_argument("--per_gpu_train_rl_batch_size", default=20, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_qa_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for QA training.")
    parser.add_argument("--with_answer", action="store_true", help="Whether to add answers into features of input of RL model..")

    # For RL agent
    parser.add_argument("--add_marginal_info", action="store_true",
                        help="Whether not to add marginal info to RL model")
    parser.add_argument(
        "--RL_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained RL estimator",
    )
    parser.add_argument(
        "--rl_num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--rl_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--rl_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Set this flag if you are using gradient checkpointing"
    )
    parser.add_argument("--rl_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--rl_weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--rl_warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--rl_adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--rl_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # For reward enviroment
    parser.add_argument(
        "--qa_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input dev file (target annotations) to provide feedback for QVE training. ",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--reward_type", default="em", type=str, help="reward type: exact/f1/loss")
    parser.add_argument("--qa_learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--score_type",default=None,type=str,help="the metric name of GPTScore")
    parser.add_argument("--use_ist",action="store_true",help="Whether to use the instruction in prompts")
    parser.add_argument("--use_demo",action="store_true",help="Whether to use the demonstrations in prompts")
    parser.add_argument("--key",default="",type=str,help='')


    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--selected_qapairs_percentage", default=0.6, type=float, help="how many questions to select?"
    )

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )


    args = parser.parse_args()
    '''
    if args.enviroment_type == "gen":
        parser = pl.Trainer.add_argparse_args(parser)
        parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
        args = parser.parse_args()
    '''

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.gpus = 1
    # args.n_gpu = 1
    return args

def train(RL_agent,dataset_rl,args,enviroment):
    if args.rl_max_steps > 0:
        train_sampler_rl = RandomSampler(dataset_rl.dataset, replacement=True,
                                         num_samples=args.train_batch_size_rl * args.rl_max_steps)
    else:
        train_sampler_rl = RandomSampler(dataset_rl.dataset)
    train_dataloader_rl = DataLoader(dataset_rl.dataset, sampler=train_sampler_rl, batch_size=args.train_batch_size_rl)
    print(len(train_dataloader_rl))
    total_train_batch_size_rl = (
            args.train_batch_size_rl
            * args.rl_gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset_rl.examples))
    logger.info("  Num Epochs = %d", RL_agent.num_train_epochs)
    logger.info("  Instantaneous batch size per device RL = %d", args.per_gpu_train_rl_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) RL = %d",total_train_batch_size_rl)
    logger.info("  Gradient Accumulation steps = %d", args.rl_gradient_accumulation_steps)
    logger.info("  Total optimization steps RL = %d", RL_agent.t_total_rl)

    global_step = 0
    epochs_trained = 0

    tr_loss_qa = torch.tensor(0.0).to(torch.device("cuda:0"))
    #print(tr_loss_qa)
    #tr_loss_rl = torch.tensor(0.0).to(args.device)
    tr_loss_rl = torch.tensor(0.0).to(torch.device("cuda:0"))
    #print(tr_loss_rl)
    #tr_reward = torch.tensor(0.0).to(args.device)
    tr_reward = torch.tensor(0.0).to(torch.device("cuda:0"))
    #tr_qa_acc = torch.tensor(0.0).to(args.device)
    tr_qa_acc = torch.tensor(0.0).to(torch.device("cuda:0"))


    best_reward = -100000
    lowest_loss = 100000

    logging_qa_loss_scalar, logging_qve_loss_scalar, logging_reward_scalar, logging_qa_acc_scalar = 0.0, 0.0, 0.0, 0.0
    RL_agent.train_init()

    baseline_performance = enviroment.get_reward()
    logger.info("Calculate %s baseline reward %f",args.reward_type,baseline_performance)

    if args.enviroment_type != "llm" and args.enviroment_type != 'str':
        init_state_dict = copy.deepcopy(enviroment.model).state_dict()
        enviroment.model.zero_grad()
    else:
        init_state_dict = None

    RL_agent.model.zero_grad()
    train_pbar = trange(epochs_trained, int(np.ceil(RL_agent.num_train_epochs)), desc="Epoch")

    for epoch in range(epochs_trained,int(np.ceil(RL_agent.num_train_epochs))):
        epoch_iterator = train_dataloader_rl
        epoch_pbar = tqdm(epoch_iterator, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            qa_values,feature_index = RL_agent.action(batch)
            qa_values = (qa_values - qa_values.min()) / (qa_values.max() - qa_values.min())

            #qa_values = qa_values.cpu().detach().numpy
            #feature_index = feature_index.cpu().detach().numpy()
            select_prob = Binomial(1,qa_values).sample()
            selected_qids = [dataset_rl.features[f].qas_id for f in feature_index]

            qa_loss, cur_performance = enviroment.reaction(selected_qids,select_prob)
            reward = cur_performance - baseline_performance
            #logger.info(f"reward is %f,",reward)
            if qa_loss is not None:
                tr_loss_qa += qa_loss
            tr_reward += reward
            tr_qa_acc += cur_performance

            rl_loss = RL_agent.update(step,select_prob,qa_values,reward)
            #print(rl_loss)
            #print(tr_loss_rl)
            tr_loss_rl += rl_loss

            if (step + 1) % args.rl_gradient_accumulation_steps == 0 or(
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.rl_gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):
                RL_agent.step_on()
                #dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                #debug(RL_agent,dataset_rl,dev_batch_size)
                global_step += 1
                epoch = epoch + (step + 1) / len(epoch_iterator)

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    tr_loss_qa_scalar = tr_loss_qa.item()
                    tr_loss_qve_scalar = tr_loss_rl.item()
                    tr_qa_acc_scalar = tr_qa_acc.item()
                    tr_reward_scalar = tr_reward.item()
                    logs["qa_loss"] = (tr_loss_qa_scalar - logging_qa_loss_scalar) / args.logging_steps
                    logs["eval_qa_current"] = (tr_qa_acc_scalar - logging_qa_acc_scalar) / args.logging_steps
                    logs["reward"] = (tr_reward_scalar - logging_reward_scalar) / args.logging_steps
                    logs["rl_loss_total"] = (tr_loss_qve_scalar - logging_qve_loss_scalar) / args.logging_steps
                    logs['eval_qa_baseline'] = baseline_performance
                    logs['num of all questions'] = len(select_prob)
                    logs['num of selected questions'] = select_prob.sum().item()


                    logging_qa_loss_scalar = tr_loss_qa_scalar
                    logging_qve_loss_scalar = tr_loss_qve_scalar
                    logging_reward_scalar = tr_reward_scalar
                    logging_qa_acc_scalar = tr_qa_acc_scalar

                    logger.info(logs)
                    save_flag = False
                    if logs['reward'] > best_reward:
                        best_reward = logs['reward']
                        output_dir = os.path.join(args.rl_output_dir,"checkpoint-best-reward")
                        save_flag = True

                    if logs["qa_loss"] < lowest_loss:
                        lowest_loss = logs["qa_loss"]
                        output_dir = os.path.join(args.rl_output_dir, "checkpoint-best-loss")
                        save_flag = True

                    if save_flag:
                        # Take care of distributed/parallel training
                        RL_agent.save(output_dir,save_opt_sche = False)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))

                        logger.info("Step: %d, Saving qve_model checkpoint to %s", global_step, output_dir)

                enviroment.reset(init_state_dict)

            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.rl_output_dir, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                RL_agent.save(output_dir, save_opt_sche=True)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Step: %d, Saving qve_model checkpoint to %s", global_step, output_dir)

            if (global_step == 1500 and 1500 % args.save_steps != 0) or (global_step == 500 and 500 % args.save_steps != 0):
                output_dir = os.path.join(args.rl_output_dir, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                RL_agent.save(output_dir, save_opt_sche=True)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Step: %d, Saving qve_model checkpoint to %s", global_step, output_dir)


            epoch_pbar.update(1)
            if args.rl_max_steps > 0 and global_step >= args.rl_max_steps:
                break

        epoch_pbar.close()
        train_pbar.update(1)

        if args.rl_max_steps > 0 and global_step >= args.rl_max_steps:
            break

    train_pbar.close()
    logger.info("\n\nTraining completed.\n\n")

def debug(RL_agent,dataset_rl,dev_batch_size):
    data_sampler = SequentialSampler(dataset_rl.dataset)
    dataloader = DataLoader(dataset_rl.dataset, sampler=data_sampler, batch_size=dev_batch_size)
    # Eval!
    logger.info("***** Running Question Value Estimation *****")
    logger.info("  Num examples = %d", len(dataset_rl.dataset))
    logger.info("  Batch size = %d", dev_batch_size)

    all_qa_values = []
    for batch in tqdm(dataloader, desc="Estimation"):
        qa_values, feature_index = RL_agent.action(batch,infer=True)
        all_qa_values.append(qa_values)

    all_qa_values = torch.cat(all_qa_values)
    all_qa_values = all_qa_values.cpu().detach().numpy()

    qid2feature = {}
    for ii, feature in enumerate(dataset_rl.features):
        if feature.qas_id not in qid2feature:
            qid2feature[feature.qas_id] = [ii]
        else:
            qid2feature[feature.qas_id].append(ii)

    qid2qv = {}
    for qid, fids in qid2feature.items():
        qid2qv[qid] = max(all_qa_values[fids])


    sorted_dict = dict(sorted(qid2qv.items(), key=lambda x: x[1], reverse=True))
    for i,key in enumerate(sorted_dict.keys()):
        print(key+ ' ' + str(sorted_dict[key]))
        if i > 10:
            break
    return

def estimation(RL_agent,dataset_rl,dev_batch_size,args,dataset_type):
    data_sampler = SequentialSampler(dataset_rl.dataset)
    dataloader = DataLoader(dataset_rl.dataset,sampler=data_sampler,batch_size=dev_batch_size)
    # Eval!
    logger.info("***** Running Question Value Estimation *****")
    logger.info("  Num examples = %d", len(dataset_rl.dataset))
    logger.info("  Batch size = %d", dev_batch_size)

    all_qa_values = []
    for batch in tqdm(dataloader, desc="Estimation"):
        qa_values,feature_index = RL_agent.action(batch,infer=True)
        all_qa_values.append(qa_values)

    all_qa_values = torch.cat(all_qa_values)
    all_qa_values = all_qa_values.cpu().detach().numpy()

    qid2feature = {}
    for ii, feature in enumerate(dataset_rl.features):
        if feature.qas_id not in qid2feature:
            qid2feature[feature.qas_id] = [ii]
        else:
            qid2feature[feature.qas_id].append(ii)

    qid2qv = {}
    for qid, fids in qid2feature.items():
        qid2qv[qid] = max(all_qa_values[fids])

    filtered_id_list = list(dict(sorted(qid2qv.items(), key=lambda x: x[1], reverse=True)).keys())[
                       :int(len(qid2qv) * args.selected_qapairs_percentage)]
    #filtered_id_list = list(dict(sorted(qid2qv.items(), key=lambda x: x[1], reverse=True)).keys())[:10]

    sorted_dict = dict(sorted(qid2qv.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(args.rl_output_dir, "debug.txt"), 'w') as f:
        for key in sorted_dict.keys():
            f.write(key + ' ' + str(sorted_dict[key]) + '\n')
        f.close()

    data_json = json.load(open(args.QAG_file, 'r'))
    if dataset_type == "ext":
        new_passages_train = []
        for passages in data_json['data']:
            new_paras_train = []

            for para in passages['paragraphs']:
                context = para['context']
                new_qas_train = []

                for qa in para['qas']:
                    if qa['id'] in filtered_id_list:
                        new_qas_train.append(qa)

                if len(new_qas_train) > 0:
                    new_paras_train.append({'context': context, 'qas': new_qas_train})

            if len(new_paras_train) > 0:
                new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

        filtered_data_json = {'data': new_passages_train, 'version': data_json['version']}

        total = 0
        context_num = 0
        for paras in data_json['data']:
            for para in paras['paragraphs']:
                context_num += 1
                qa_num = len(para['qas'])
                total += qa_num
        logger.info('Before filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

        total = 0
        context_num = 0
        for paras in filtered_data_json['data']:
            for para in paras['paragraphs']:
                context_num += 1
                qa_num = len(para['qas'])
                total += qa_num
        logger.info('After filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

    else:
        filted_data = []
        questions = []
        contexts = []
        answers = []
        logger.info('Before filtering: Train QA Num: %d' % len(data_json['data']))
        for data in data_json['data']:
            if data["qid"] in filtered_id_list:
                filted_data.append(data)
                questions.append(data["question"])
                contexts.append(data["context"])
                answers.append(data["answer"])

        logger.info('After filtering: Train QA Num: %d' % len(filted_data))
        filtered_data_json = {"data":filted_data}
        with open(os.path.join(args.rl_output_dir, "filtered.source"),'w') as f:
            for question, context in zip(questions,contexts):
                f.write(question + '<SEP>' + context + '\n')
            f.close()
        with open(os.path.join(args.rl_output_dir, "filtered.target"),'w') as f:
            for answer in answers:
                f.write(answer + '\n')
            f.close()


    json.dump(filtered_data_json, open(os.path.join(args.rl_output_dir, "filtered_qa.json"), 'w'))

    return





def main():
    args = get_argument()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s",
        args.device,
        args.n_gpu,
    )
    # Set seed
    set_seed(args)

    if args.dataset_name == "squad" or args.dataset_name == "NaturalQuestionsShort" or args.dataset_name == "NewsQA" or args.dataset_name == "TriviaQA-web":
        data_type = "json"
        dataset_type = "ext"
    else:
        data_type = "txt"
        dataset_type = "gen"

    # for dataset

    tokenizer = AutoTokenizer.from_pretrained(
        args.rl_tokenizer_name if args.rl_tokenizer_name else args.RL_model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.rl_cache_dir if args.rl_cache_dir else None,
        use_fast=False
    )
    ## adding special tokens
    flag_add = False

    if '<ANS>' not in tokenizer.additional_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<ANS>', '<NULL_ANS>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info("Adding Special Tokens: %s", special_tokens_dict)
        flag_add = True

    dataset_rl = RLDataset(args.QAG_file,args.dataset_name,data_type,tokenizer,
                           max_seq_length=args.max_seq_length,doc_stride=args.doc_stride,
                           max_query_length=args.max_query_length,overwrite_cache=args.overwrite_cache)
    args.train_batch_size_rl = args.per_gpu_train_rl_batch_size * max(1, args.n_gpu)
    if len(dataset_rl.dataset) % args.train_batch_size_rl == 0:
        train_dataloader_len = len(dataset_rl.dataset) // args.train_batch_size_rl
    else:
        train_dataloader_len = len(dataset_rl.dataset) // args.train_batch_size_rl + 1

    # For RL agent
    config = AutoConfig.from_pretrained(
        args.RL_model_name_or_path,
        num_labels=2,
        gradient_checkpointing=args.gradient_checkpointing,
        cache_dir=args.rl_cache_dir if args.rl_cache_dir else None,
    )
    config.marginal = args.add_marginal_info
    RL_model = BertForSequenceClassification.from_pretrained(
        args.RL_model_name_or_path,
        from_tf=bool(".ckpt" in args.RL_model_name_or_path),
        config=config,
        cache_dir=args.rl_cache_dir if args.rl_cache_dir else None,
    )

    if flag_add:
        RL_model.resize_token_embeddings(len(tokenizer))

    RL_model.to(args.device)
    if args.n_gpu > 1:
        RL_model = torch.nn.DataParallel(RL_model)

    RL_agent = RLAgent_BERT(RL_model,train_dataloader_len,args.device,args.rl_gradient_accumulation_steps,args.rl_max_steps,
                            args.rl_num_train_epochs,args.rl_learning_rate,args.rl_max_grad_norm,args.rl_weight_decay,args.rl_adam_epsilon,args.rl_warmup_steps)
    logger.info("Training/evaluation parameters %s", args)

    #dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #estimation(RL_agent, dataset_rl, dev_batch_size, args, dataset_type)

    if args.do_rl_train:
        # For reward enviroment
        #args.reward_type = args.reward_type.lower()
        if args.enviroment_type == "ext":
            config = AutoConfig.from_pretrained(
                args.qa_model_name_or_path,
                gradient_checkpointing=args.gradient_checkpointing,
                cache_dir=args.rl_cache_dir if args.rl_cache_dir else None,
            )

            qa_model = BertForQuestionAnswering.from_pretrained(
                args.qa_model_name_or_path,
                config=config,
                cache_dir=args.rl_cache_dir if args.rl_cache_dir else None,
            )
            qa_model.to(args.device)
            Agent_dataset = SquadDataset(args.QAG_file,tokenizer,version="1.0",istrain=True,max_seq_length=args.max_seq_length,doc_stride=args.doc_stride,
                                   max_query_length=args.max_query_length,overwrite_cache=args.overwrite_cache)
            dev_dataset = SquadDataset(args.dev_file,tokenizer,version="1.0",istrain=False,max_seq_length=args.max_seq_length,doc_stride=args.doc_stride,
                                   max_query_length=args.max_query_length,overwrite_cache=args.overwrite_cache)
            args.train_batch_size_qa = args.per_gpu_train_qa_batch_size * max(1,args.n_gpu)
            qa_performance_environment = QA_Performance_Enviroment(args.reward_type, qa_model, Agent_dataset,
                                                                   dev_dataset, args.train_batch_size_qa,
                                                                   args.per_gpu_eval_batch_size, args.qa_learning_rate,
                                                                   args.n_gpu, args.device, args.rl_max_grad_norm,
                                                                   args.rl_weight_decay, args.rl_adam_epsilon)
        elif args.enviroment_type == "gen":
            SummarizationModule.istest = False
            args.config_name = args.model_name_or_path
            model = SummarizationModule(args)
            if args.qa_model_name_or_path is not None:
                model = model.load_from_checkpoint(
                    args.qa_model_name_or_path,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    freeze_encoder=args.freeze_encoder,
                    max_epochs=args.max_epochs,
                    train_batch_size=args.train_batch_size,
                    eval_beams=args.eval_beams,
                    eval_batch_size=args.eval_batch_size,
                    fuse_num=args.fuse_num,
                    type_embedding=args.type_embedding,
                )
            model_device = torch.device("cuda:0")
            #model_device = args.device
            #model.to(model_device)
            dev_dataset = RLDataset(args.dev_file,args.dataset_name,data_type,tokenizer,
                           max_seq_length=args.max_seq_length,doc_stride=args.doc_stride,
                           max_query_length=args.max_query_length,overwrite_cache=True,need_features=False)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            qa_performance_environment = GEN_QA_Performance_Enviroment(args.reward_type,model,args.rl_output_dir,
                                                                       dataset_rl.examples,dev_dataset.examples,args,
                                                                       args.eval_batch_size,tokenizer,model_device)
        elif args.enviroment_type == "llm":
            metric2checkpoint = {
                "opt125m_score": "facebook/opt-125m",
                "opt350m_score": "facebook/opt-350m",
                "opt1_3B_score": "facebook/opt-1.3b",
                "opt2_7B_score": "facebook/opt-2.7b",
                "opt6_7B_score": "facebook/opt-6.7b",
                "opt13B_score": "facebook/opt-13b",
                "opt66B_score": "facebook/opt-66b",
                "gpt2_medium_score": "gpt2-medium",
                "gpt2_large_score": "gpt2-large",
                "gpt2_xl_score": "gpt2-xl",
                "gptJ6B_score": "EleutherAI/gpt-j-6b",
                "flan_small_score": "google/flan-t5-small",
                "flan_base_score": "google/flan-t5-base",
                "flan_large_score": "google/flan-t5-large",
                "flan_xl_score": "google/flan-t5-xl",
                "flan_xxl_score": "google/flan-t5-xxl",
                'llama2_score':"meta-llama/Llama-2-7b-hf",
                'llama2_chat_score':"meta-llama/Llama-2-7b-chat-hf"
            }
            checkpoint = args.qa_model_name_or_path
            if args.score_type in  ["flan_small_score", "flan_base_score", "flan_large_score","flan_xl_score", "flan_xxl_score"]:
                Scorer = FLANScorer(device = args.device,checkpoint=checkpoint)
            elif args.score_type in  [
                "opt125m_score", "opt350m_score", "opt1_3B_score",
                "opt2_7B_score", "opt6_7B_score", "opt13B_score", "opt30B_score", "opt66B_score",
                "gpt2_medium_score", "gpt2_large_score", "gpt2_xl_score", "gptJ6B_score",'llama2_score','llama2_chat_score'
            ]:
                Scorer = OPTScorer(device=args.device,checkpoint=checkpoint)
            else:
                Scorer = GPT3Scorer(model_name = checkpoint, api_key= args.api_key)

            dev_dataset = RLDataset(args.dev_file, args.dataset_name, data_type, tokenizer,
                                    max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
                                    max_query_length=args.max_query_length, overwrite_cache=True, need_features=False)
            '''
            qa_performance_environment = LLMEnviroment(args.reward_type,Scorer,dataset_rl.examples,random.sample(dev_dataset.examples,1),
                                                       "./data/llm_instruction_design.json",args.device,use_ist=args.use_ist,
                                                       use_demo=args.use_demo)
            '''
            qa_performance_environment = OptLLMEnviroment(args.reward_type,Scorer,dataset_rl.examples,random.sample(dev_dataset.examples,1),
                                                       "./data/llm_instruction_design.json",args.device,use_ist=args.use_ist,
                                                       use_demo=args.use_demo)

        else:
            qa_performance_environment = StrictEnvironment(args.qa_model_name_or_path,args.device)



        train(RL_agent,dataset_rl,args,qa_performance_environment)

        logger.info("Saving model checkpoint to %s", args.rl_output_dir)

        RL_agent.save(args.rl_output_dir,save_opt_sche=False)
        tokenizer.save_pretrained(args.rl_output_dir)
        torch.save(args, os.path.join(args.rl_output_dir,"training_args.bin"))

    if args.do_rl_estimation:
        if args.do_rl_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = args.rl_output_dir
        else:
            logger.info("Loading checkpoint %s for evaluation", args.RL_model_name_or_path)
            checkpoints = args.RL_model_name_or_path

        config = AutoConfig.from_pretrained(
            checkpoints,
            gradient_checkpointing=False,
        )
        RL_model = BertForSequenceClassification.from_pretrained(checkpoints, config=config)

        RL_model.to(args.device)
        if args.n_gpu > 0:
            RL_model = torch.nn.DataParallel(RL_model)

        RL_agent = RLAgent_BERT(RL_model, train_dataloader_len, args.device, args.rl_gradient_accumulation_steps,
                                args.rl_max_steps,
                                args.rl_num_train_epochs, args.rl_learning_rate, args.rl_weight_decay, args.rl_adam_epsilon,
                                args.rl_warmup_steps)

        dataset_rl = RLDataset(args.QAG_file, args.dataset_name, data_type, tokenizer,
                               max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
                               max_query_length=args.max_query_length, overwrite_cache=args.overwrite_cache)

        dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        estimation(RL_agent,dataset_rl,dev_batch_size,args,dataset_type)

if __name__ == "__main__":
    main()





