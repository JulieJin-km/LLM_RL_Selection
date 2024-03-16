from transformers import BertModel,BertPreTrainedModel
from transformers.file_utils import add_start_docstrings_to_model_forward,add_code_sample_docstrings
from transformers.models.bert.modeling_bert import (BERT_INPUTS_DOCSTRING,
                                                    _CHECKPOINT_FOR_DOC,_CONFIG_FOR_DOC,QuestionAnsweringModelOutput)
from transformers import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.distributions.binomial import Binomial
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler,DataLoader,RandomSampler
from collections import Counter
from rouge_score import rouge_scorer,scoring
import os
from pathlib import Path
import pytorch_lightning as pl
from tqdm import tqdm
import copy
import logging
import json
import numpy as np
import random

from QG_models import SummarizationModule
from QG_callback import get_early_stopping_callback,Seq2SeqLoggingCallback,get_checkpoint_callback
from lightning_base import generic_train,logger as train_logger

_TOKENIZER_FOR_DOC = "BertTokenizer"
logger = logging.getLogger(__name__)

def use_task_specific_params(model,task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task,{})
        logger.info(f"using task specific params for {task}:{pars}")
        model.config.update(pars)

def chunks(lst,n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

class BertForQuestionAnswering(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    try:
        @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            tokenizer_class=_TOKENIZER_FOR_DOC,
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=QuestionAnsweringModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                input_values=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
        ):
            r"""
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                # weighted training
                # taking selection probs into consideration
                if input_values is None:
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                else:
                    while input_values.sum() == 0:
                        # 如果此时没有选的，那就随机挑
                        input_values = 0.5 * torch.ones_like(input_values)
                        input_values = Binomial(1,input_values).sample()
                    start_loss = loss_fct(start_logits,start_positions) * input_values
                    start_loss = start_loss.sum() / input_values.sum()
                    end_loss = loss_fct(end_logits,end_positions) * input_values
                    end_loss = end_loss.sum() / input_values.sum()

                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    except:
        @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=QuestionAnsweringModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                input_values=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
        ):
            r"""
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
                sequence are not taken into account for computing the loss.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

                # weighted training
                # taking selection probs into consideration
                if input_values is None:
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                else:
                    while input_values.sum() == 0:
                        # 如果此时没有选的，那就随机挑
                        input_values = 0.5 * torch.ones_like(input_values)
                        input_values = Binomial(1, input_values).sample()
                    start_loss = loss_fct(start_logits, start_positions) * input_values
                    start_loss = start_loss.sum() / input_values.sum()
                    end_loss = loss_fct(end_logits, end_positions) * input_values
                    end_loss = end_loss.sum() / input_values.sum()

                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

def get_optimizer(model,weight_decay,adam_epsilon,learning_rate):
    """
        Setup the optimizer and the learning rate scheduler.
        """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon,
    )
    return optimizer

class Enviroment(object):
    def reset(self,*input):
        raise Exception("Not Implemented yet")

    def get_reward(self,*input):
        raise Exception("Not Implemented yet")

    def reaction(self,*input):
        return self.get_reward(input)


class QA_Performance_Enviroment(Enviroment):
    def __init__(self,reward_type,Model,agent_dataset,dev_dataset,train_batch,per_gpu_eval_batch,lr,n_gpu,device,max_grad_norm,weight_decay,adam_epsilon,data_type = "ext"):
        self.reward_type = reward_type
        self.data_type = data_type
        self.model = Model
        self.agent_dataset = agent_dataset
        self.dev_dataset = dev_dataset
        self.train_batch = train_batch
        self.per_gpu_eval_batch = per_gpu_eval_batch
        self.lr = lr
        self.n_gpu = n_gpu
        self.dev_batch_size = self.per_gpu_eval_batch * max(1, self.n_gpu)
        self.device = device
        self.max_grad_norm = max_grad_norm
        if self.data_type == "ext":
            self.optimizer = get_optimizer(self.model,weight_decay,adam_epsilon,self.lr)
            self.qid2features = self.construct_qid_features()


    def construct_qid_features(self):
        qid2features = {}
        for feature in self.agent_dataset.features:
            if feature.qas_id not in qid2features:
                qid2features[feature.qas_id] = [feature]
            else:
                qid2features[feature.qas_id].append(feature)

        return qid2features

    def construct_inputs(self,selected_qids,select_probs):
        if self.data_type == "ext":
            features = []
            probs = []
            for i,qid in enumerate(selected_qids):
                selected = self.qid2features[qid]
                prob = select_probs[i]
                assert len(selected) > 0
                features.extend(self.qid2features[qid])
                probs.extend([prob] * len(selected))
            probs = torch.tensor(probs)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                probs
            )
            return dataset
        return

    def train_ext_qa(self,train_dataloader):
        total_loss = 0
        for step_i, batch_i in enumerate(train_dataloader):
            self.model.train()
            batch_i = tuple(t.to(self.device) for t in batch_i)
            inputs = {
                "input_ids": batch_i[0],
                "attention_mask": batch_i[1],
                "token_type_ids": batch_i[2],
                "start_positions": batch_i[3],
                "end_positions": batch_i[4],
                "input_values": batch_i[5],
            }
            outputs = self.model(**inputs)
            loss = outputs[0]
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()

        return total_loss / len(train_dataloader)

    def get_reward(self):
        self.model.eval()
        dev_sampler = SequentialSampler(self.dev_dataset.dataset)
        dev_dataloader = DataLoader(self.dev_dataset.dataset,sampler=dev_sampler,batch_size=self.dev_batch_size)
        total_reward = 0
        for dev_batch in dev_dataloader:
            dev_batch = tuple(t.to(self.device) for t in dev_batch)
            with torch.no_grad():
                if self.reward_type == "loss":
                    inputs = {
                        "inputs_ids": dev_batch[0],
                        "attention_mask": dev_batch[1],
                        "token_type_ids": dev_batch[2],
                        "start_positions": dev_batch[3],
                        "end_positions": dev_batch[4]
                    }
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    if self.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    total_reward -= loss * 10

                elif self.reward_type == "em":
                    inputs = {
                        "input_ids": dev_batch[0],
                        "attention_mask": dev_batch[1],
                        "token_type_ids": dev_batch[2],
                    }
                    outputs = self.model(**inputs)
                    pred_start_indices = torch.argmax(outputs[0], dim=1)
                    pred_end_indices = torch.argmax(outputs[1], dim=1)
                    start_acc = pred_start_indices == dev_batch[3]
                    end_acc = pred_end_indices == dev_batch[4]
                    acc_num = torch.logical_and(start_acc, end_acc).sum()
                    total_reward += acc_num

                elif self.reward_type == "f1":
                    def f1_score(prediction, ground_truth):
                        prediction = prediction.cpu().numpy().tolist()
                        ground_truth = ground_truth.cpu().numpy().tolist()
                        common = Counter(prediction) & Counter(ground_truth)
                        num_same = sum(common.values())
                        if num_same == 0:
                            return 0
                        precision = 1.0 * num_same / len(prediction)
                        recall = 1.0 * num_same / len(ground_truth)
                        f1 = (2 * precision * recall) / (precision + recall)
                        return f1

                    inputs = {
                        "input_ids": dev_batch[0],
                        "attention_mask": dev_batch[1],
                        "token_type_ids": dev_batch[2],
                    }
                    outputs = self.model(**inputs)
                    pred_start_indices = torch.argmax(outputs[0], dim=1)
                    pred_end_indices = torch.argmax(outputs[1], dim=1)
                    for i in range(len(dev_batch[0])):
                        pred_start = pred_start_indices[i]
                        pred_end = pred_end_indices[i]
                        gt_start = dev_batch[4][i]
                        gt_end = dev_batch[5][i]
                        prediction = dev_batch[0][i][gt_start:gt_end + 1]
                        ground_truth = dev_batch[0][i][pred_start:pred_end + 1]
                        total_reward += f1_score(prediction, ground_truth)

        return 100.0 * total_reward / len(self.dev_dataset.dataset)

    def reaction(self,selected_qids,select_probs):
        inputs = self.construct_inputs(selected_qids,select_probs)
        train_QA_sampler = RandomSampler(inputs)
        train_QA_dataloader = DataLoader(inputs, sampler=train_QA_sampler, batch_size=self.train_batch)
        qa_loss = self.train_ext_qa(train_QA_dataloader)

        cur_performance = self.get_reward()
        return qa_loss,cur_performance

    def reset(self,init_statedict):
        self.model.load_state_dict(init_statedict)


class GEN_QA_Performance_Enviroment(Enviroment):
    def __init__(self,reward_type,model,output_dir,agent_examples,dev_examples,args,dev_batch_size,tokenizer,model_device):
        self.reward_type = reward_type
        self.model = model
        self.output_dir = output_dir
        self.agent_examples = agent_examples
        self.dev_examples = dev_examples
        self.agent_qid2examples = self.get_qid2examples(self.agent_examples)
        self.dev_qid2examples = self.get_qid2examples(self.dev_examples)
        self.args = args
        self.dev_batch_size = dev_batch_size
        self.tokenizer = tokenizer
        self.device = model_device
        self.args.device = self.device
        self.args.do_train = False
        if self.args.early_stopping_patience >= 0:
            es_callback = get_early_stopping_callback(self.model.val_metric, self.args.early_stopping_patience)
        else:
            es_callback = False
        lower_is_better = self.args.val_metric == "loss"
        train_logger = True
        trainer: pl.Trainer = generic_train(
            self.model,
            self.args,
            logging_callback=Seq2SeqLoggingCallback(),
            checkpoint_callback=get_checkpoint_callback(
                self.args.output_dir, self.model.val_metric, self.args.save_top_k, lower_is_better
            ),
            early_stopping_callback=es_callback,
            logger=train_logger,
        )
        self.trainer = trainer


    def get_qid2examples(self,dataset):
        qid2examples = {}
        for example in dataset:
            qid2examples[example.qas_id] = example
        return qid2examples

    def train_qa(self):
        SummarizationModule.istest = False
        self.trainer.fit(self.model,val_dataloaders=None)
        return

    def get_reward(self):
        # test
        # method1: can't
        # results = trainer.test()
        # method2
        self.model.to(self.device)
        SummarizationModule.istest = True
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        input_for_qa_model = []
        label_for_qa_model = []
        for example in self.dev_examples:
            question = example.question_text
            context = example.context_text
            answer = example.answer_text
            input_for_qa_model.append(question + '<SEP>' + context)
            label_for_qa_model.append(answer)
        assert len(input_for_qa_model) == len(label_for_qa_model)
        # begin to eval the model
        self.model.eval()
        self.tokenizer.add_special_tokens({'sep_token': '<SEP>'})
        logger.info('#############################################')
        logger.info('# tokenizer.all_special_tokens =', self.model.tokenizer.all_special_tokens)
        logger.info('# tokenizer.all_special_ids =', self.tokenizer.all_special_ids)
        logger.info('#############################################')
        use_task_specific_params(self.model,"summarization")
        output_for_qa_model = []
        with torch.no_grad():
            for examples_chunk in tqdm(list(chunks(input_for_qa_model, self.dev_batch_size))):
                examples_chunk = [text for text in examples_chunk]
                batch = self.model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(
                    self.device)
                if len(batch.input_ids[0]) > 1024:
                    end_token = self.model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:].to(self.device)
                    # 截断
                    input_ids = torch.cat((batch.input_ids[0][:1022], end_token), 0).unsqueeze(0)
                    batch.input_ids = input_ids

                summaries = self.model.generate(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                    )
                dec = self.model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for hypothesis in dec:
                    output_for_qa_model.append(hypothesis)

        for i in range(len(output_for_qa_model)):
            output_for_qa_model[i] = output_for_qa_model[i].replace("\n", "").split(".")[0].strip(" .") + " ."
            score = scorer.score(label_for_qa_model[i], output_for_qa_model[i])
            aggregator.add_scores(score)

        assert len(input_for_qa_model) == len(output_for_qa_model)
        results = aggregator.aggregate()
        rewards = {}
        rewards['rouge1'] = results['rouge1'].mid.fmeasure
        rewards['rouge2'] = results['rouge2'].mid.fmeasure
        rewards['rougeL'] = results['rougeL'].mid.fmeasure

        reward = rewards[self.reward_type]

        return reward

    def construct_inputs(self, selected_qids):
        answers = []
        questions = []
        contexts = []
        for qid in selected_qids:
            example = self.agent_qid2examples[qid]
            answers.append(example.answer_text)
            questions.append(example.question_text)
            contexts.append(example.context_text)
        with open(os.path.join(self.args.data_dir,"train.source"),'w') as f:
            for question,context in zip(questions,contexts):
                f.write(question + '<SEP>' + context + '\n')
            f.close()
        with open(os.path.join(self.args.data_dir,"train.target"),'w') as f:
            for answer in answers:
                f.write(answer + '\n')
            f.close()
        return

    def reaction(self,selected_qids,selected_probs):
        qids = []
        for i in range(len(selected_probs)):
            if selected_probs[i] > 0:
                qids.append(selected_qids[i])
        self.construct_inputs(qids)
        self.train_qa()
        reward = self.get_reward()
        try:
            os.remove(self.args.output_dir + '/epoch=0.ckpt')
        except:
            print("Nothing to delete.")
        return None, reward

    def reset(self,init_state_dict):
        self.model.load_state_dict(init_state_dict)
        return


class LLMEnviroment(Enviroment):
    def __init__(self,reward_type,Scorer,agent_examples,dev_examples,asp_path,device,use_ist = True,use_demo = True):
        self.reward_type = reward_type
        self.Scorer = Scorer
        self.agent_examples = agent_examples
        self.agent_qid2examples = self.get_qid2examples(self.agent_examples)
        self.demos = dev_examples
        self.asp_dfs = self.read_asp(asp_path)
        self.use_ist = use_ist
        self.use_demo = use_demo
        self.device = device

    def get_qid2examples(self,dataset):
        qid2examples = {}
        for example in dataset:
            qid2examples[example.qas_id] = example
        return qid2examples

    def read_asp(self,asp_path):
        asp_data = json.load(open(asp_path))
        asp_dfs = asp_data["asp_definition"]
        return asp_dfs

    def demo_convert(self,demos,template):
        rl_demos = []
        for demo in demos:
            context = demo.context_text
            question = demo.question_text + '?'
            answer = demo.answer_text + '.'
            t = template.replace('CCCCCC',context).replace('QQQQQQ',question).replace('AAAAAA',answer)
            rl_demos.append(t)
        return rl_demos

    def construct_input(self,selected_qids,selected_probs):
        qids = []
        for i in range(len(selected_probs)):
            if selected_probs[i] > 0:
                qids.append(selected_qids[i])
        selected_examples = []
        for qid in qids:
            selected_examples.append(self.agent_qid2examples[qid])
        return selected_examples

    def get_reward(self,datas = None):
        if datas is None:
            datas = random.sample(self.agent_examples,40)
        #asp_df_d = self.asp_dfs[self.reward_type].strip().replace(':','. ')
        asp_df_d = self.asp_dfs[self.reward_type].strip()
        asp_df_m = self.asp_dfs['MRC'].strip()
        #print('demos: ',self.demos)
        #print('asp_df:', asp_df)
        dialog_template = ["Conversation: human asked, CCCCCC QQQQQQ. AI answered AAAAAA",
                           "Conversation: human: CCCCCCC QQQQQQ. AI: AAAAAA",
                           "Conversation: human asked, CCCCCC Based on the context, QQQQQQ. AI answered AAAAAA",
                           "Conversation: human: CCCCCCC Based on the context, QQQQQQ. AI: AAAAAA",
                           "Conversation: human said, answer the question based on the given text. The question is QQQQQQ The text is CCCCCC. AI answered AAAAAA"]
        mrc_template = ["Question: QQQQQQ\nContext: CCCCCC"]
        m_template = mrc_template[0]
        d_template = dialog_template[4]
        demo_template1 = m_template + '\nAnswer:AAAAAA'
        demo_template2 = d_template + '\nAnswer: Yes.'
        rl_demos = self.demo_convert(self.demos,demo_template1)
        demos2 = self.demo_convert(self.demos,demo_template2)
        if self.use_ist and self.use_demo:
            demos_str = "\n".join(rl_demos)
            prefix_d = asp_df_d + '\n' + demos_str + '\n'
            demos_mrc = "\n".join(demos2)
            prefix_m = asp_df_m + '\n' + demos_mrc + '\n'
        elif self.use_ist and not self.use_demo:
            prefix_d = asp_df_d + '\n'
            prefix_m = asp_df_m + '\n'
        elif not self.use_ist and not self.use_demo:
            prefix_d = ''
            prefix_m = ''
        scores = []
        for data in datas:
            context = data.context_text
            question = data.question_text
            if question[-1] != '?':
                question = question + '?'
            answer = data.answer_text + '.'
            if answer[-1] != '.':
                answer = answer + '.'
            # mrc score
            '''
            inputs = m_template.replace('CCCCCC', context).replace('QQQQQQ', question).replace('AAAAAA',
                                                                                               answer) + '\nAnswer:\n'
            # print(len(prefix + inputs))
            lens = len(prefix_m + inputs)
            if lens > self.Scorer.max_length:
                # logger.info("%d is out of max length limitation!",lens)
                prefix_input = asp_df_m + '\n' + inputs
            else:
                prefix_input = prefix_m + inputs
            output = self.Scorer.score([prefix_input], [answer], prompt_text="\nAnswer: ")
            # score = np.array(output)
            score1 = output[0]
            '''
            # dialog score
            inputs = d_template.replace('CCCCCC',context).replace('QQQQQQ',question).replace('AAAAAA',answer) + '\nAnswer:\n'
            #print(len(prefix + inputs))
            lens = len(prefix_d + inputs)
            if lens > self.Scorer.max_length:
                #logger.info("%d is out of max length limitation!",lens)
                prefix_input = asp_df_d + '\n' + inputs
            else:
                prefix_input = prefix_d+ inputs
            output = self.Scorer.score([prefix_input],["Yes"],prompt_text = "\nAnswer: ")
            #score = np.array(output)
            score2 = output[0]

            score = score2
            scores.append(score)
        avg = sum(scores)/len(scores)
        #test = torch.tensor(1.5)
        reward = torch.tensor(avg).to(self.device)
        return reward

    def reaction(self,selected_qids,selected_probs):
        selected_examples = self.construct_input(selected_qids,selected_probs)
        reward = self.get_reward(selected_examples)
        return None, reward

    def reset(self,init_state = None):
        return


class OptLLMEnviroment(Enviroment):
    def __init__(self,reward_type,Scorer,agent_examples,dev_examples,asp_path,device,use_ist = True,use_demo = True):
        self.reward_type = reward_type
        self.Scorer = Scorer
        self.agent_examples = agent_examples
        self.agent_qid2examples = self.get_qid2examples(self.agent_examples)
        self.demos = dev_examples
        self.asp_dfs = self.read_asp(asp_path)
        self.use_ist = use_ist
        self.use_demo = use_demo
        self.device = device
        self.score_type = 0
        # 0: pure dialog
        # 1: dialog + mrc
        # 2: dialog + question
        # 3: dialog + question + mrc
        # 4: 0,1 reward

    def get_qid2examples(self,dataset):
        qid2examples = {}
        for example in dataset:
            qid2examples[example.qas_id] = example
        return qid2examples

    def read_asp(self,asp_path):
        asp_data = json.load(open(asp_path))
        asp_dfs = asp_data["asp_definition"]
        return asp_dfs

    def demo_convert(self,demos,template):
        rl_demos = []
        for demo in demos:
            context = demo.context_text
            question = demo.question_text + '?'
            answer = demo.answer_text + '.'
            t = template.replace('CCCCCC',context).replace('QQQQQQ',question).replace('AAAAAA',answer)
            rl_demos.append(t)
        return rl_demos

    def construct_input(self,selected_qids,selected_probs):
        qids = []
        for i in range(len(selected_probs)):
            if selected_probs[i] > 0:
                qids.append(selected_qids[i])
        selected_examples = []
        for qid in qids:
            selected_examples.append(self.agent_qid2examples[qid])
        return selected_examples

    def get_logscore(self, template, question, context, answer, prefix, asp_df, scorer_answer):
        inputs = template.replace('CCCCCC', context).replace('QQQQQQ', question).replace('AAAAAA',
                                                                                           answer) + '\nAnswer:\n'
        # print(len(prefix + inputs))
        lens = len(prefix + inputs)
        if lens > self.Scorer.max_length:
            # logger.info("%d is out of max length limitation!",lens)
            prefix_input = asp_df + '\n' + inputs
        else:
            prefix_input = prefix + inputs
        output = self.Scorer.score([prefix_input], [scorer_answer], prompt_text="\nAnswer: ")
        # score = np.array(output)
        score = output[0]
        return score

    def get_reward(self,datas = None):
        if datas is None:
            datas = random.sample(self.agent_examples,40)
        #asp_df_d = self.asp_dfs[self.reward_type].strip().replace(':','. ')
        asp_df_d = self.asp_dfs[self.reward_type].strip()
        asp_df_m = self.asp_dfs['MRC'].strip()
        asp_df_q = self.asp_dfs['QUE'].strip()
        #print('demos: ',self.demos)
        #print('asp_df:', asp_df)
        dialog_template = ["Conversation: Human asked, CCCCCC. QQQQQQ AI answered AAAAAA",
                           "Conversation: Human: CCCCCCC. QQQQQQ AI: AAAAAA",
                           "Conversation: Human asked, CCCCCC. Based on the context, QQQQQQ AI answered AAAAAA",
                           "Conversation: Human: CCCCCCC. Based on the context, QQQQQQ AI: AAAAAA",
                           "Conversation: Human said, answer the question based on the given text. The question is QQQQQQ The text is CCCCCC. AI answered AAAAAA"]
        mrc_template = ["Question: QQQQQQ\nContext: CCCCCC"]
        question_template = ["Conversation: Human said, generate a question based on the following text. CCCCCC. AI said QQQQQQ"]
        m_template = mrc_template[0]
        d_template = dialog_template[0]
        q_template = question_template[0]
        demo_template1 = m_template + '\nAnswer:AAAAAA'
        demo_template2 = d_template + '\nAnswer: Yes.'
        demo_template3 = q_template + '\nAnswer: Yes.'
        rl_demos = self.demo_convert(self.demos,demo_template1)
        demos2 = self.demo_convert(self.demos,demo_template2)
        demos3 = self.demo_convert(self.demos,demo_template3)
        if self.use_ist and self.use_demo:
            demos_str = "\n".join(rl_demos)
            prefix_d = asp_df_d + '\n' + demos_str + '\n'
            demos_mrc = "\n".join(demos2)
            prefix_m = asp_df_m + '\n' + demos_mrc + '\n'
            demos_ques = "\n".join(demos3)
            prefix_q = asp_df_q + '\n' + demos_ques + '\n'
        elif self.use_ist and not self.use_demo:
            prefix_d = asp_df_d + '\n'
            prefix_m = asp_df_m + '\n'
            prefix_q = asp_df_q + '\n'
        elif not self.use_ist and not self.use_demo:
            prefix_d = ''
            prefix_m = ''
            prefix_q = ''
        scores = []
        for data in datas:
            context = data.context_text
            question = data.question_text
            if question[-1] != '?':
                question = question + '?'
            answer = data.answer_text + '.'
            if answer[-1] != '.':
                answer = answer + '.'
            contexts = []
            all_context = context
            contexts.append(all_context)
            window_size = 1000
            doc_stride = 512
            i = 0
            # method1:
            if i + window_size < len(all_context):
                while i < len(all_context):
                    sub_context = context[i:i + window_size]
                    contexts.append(sub_context)
                    if i + window_size >= len(all_context):
                        break
                    i += doc_stride
            # method2:
            '''
            while i + window_size <= len(all_context):
                sub_context = context[i:i + window_size]
                contexts.append(sub_context)
                i += doc_stride
            '''

            sub_scores = []
            for context in contexts:
                if self.score_type == 0:
                    score_d = self.get_logscore(d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                    score = score_d
                elif self.score_type == 1:
                    score_d = self.get_logscore(d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                    score_m = self.get_logscore(m_template, question, context, answer, prefix_m, asp_df_m, answer)
                    score = (score_d + score_m) / 2
                elif self.score_type == 2:
                    score_q = self.get_logscore(q_template, question, context, answer, prefix_q, asp_df_q, "Yes")
                    score_d = self.get_logscore(d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                    score = (score_q + score_d) / 2
                elif self.score_type == 3:
                    score_q = self.get_logscore(q_template, question, context, answer, prefix_q, asp_df_q, "Yes")
                    score_d = self.get_logscore(d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                    score_m = self.get_logscore(m_template, question, context, answer, prefix_m, asp_df_m, answer)
                    score = (score_q + score_d + score_m) / 3


                sub_scores.append(score)

            scores.append(max(sub_scores))
        avg = sum(scores)/len(scores)
        #test = torch.tensor(1.5)
        reward = torch.tensor(avg).to(self.device)
        return reward

    def reaction(self,selected_qids,selected_probs):
        selected_examples = self.construct_input(selected_qids,selected_probs)
        reward = self.get_reward(selected_examples)
        return None, reward

    def reset(self,init_state = None):
        return


class StrictEnvironment(Enviroment):
    def __init__(self,score_file,device):
        self.score_file = score_file
        with open(score_file,'r') as f:
            self.scores = json.load(f)
            f.close()
        self.device = device

    def get_reward(self,selected_qids = None,selected_probs = None):
        if selected_qids is None:
            qids = random.sample(self.scores.keys(),40)
        else:
            qids = []
            for i in range(len(selected_probs)):
                if selected_probs[i] > 0:
                    qids.append(selected_qids[i])

        scores = []
        for qid in qids:
            scores.append(self.scores[qid])

        avg = sum(scores) / len(scores)
        reward = torch.tensor(avg).to(self.device)

        return reward

    def reaction(self,selected_qids,selected_probs):
        reward = self.get_reward(selected_qids,selected_probs)
        return None, reward

    def reset(self,init_state = None):
        return