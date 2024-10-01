
from models import SummarizationModule
from utils import use_task_specific_params


import glob
import torch
import argparse
from allennlp.predictors.predictor import Predictor
import spacy
from spacy import displacy
from typing import List,Dict
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
DEFAULT_DEVICE = '0' if torch.cuda.is_available() else 'cpu'

def chunks(lst,n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def generate_id_to_logits(model, generated_ids, src_ids, src_masks):
    pad_token_id = model.tokenizer.pad_token_id
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id,reduction="none")
    generated_ids[:,0] = 0
    gen_decoder_input_ids = shift_tokens_right(generated_ids,pad_token_id)
    labels = generated_ids
    with torch.no_grad():
        logits = model(src_ids, attention_mask = src_masks, decoder_input_ids = gen_decoder_input_ids, use_cache=False)[0]

    logprob = -loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(
        logits.shape[0], -1)
    return logprob.sum(dim=-1).cpu().numpy()


def generate_summaries_or_translations(
    model,
    examples: List[str],
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix='',
    args=None,
    **generate_kwargs,
) -> Dict:
    # save outputs to a list
    output_list = []
    use_task_specific_params(model,task)
    for examples_chunk in tqdm(list(chunks(examples,batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]

        if device == 'cpu':
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest")
        else:
            batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(
                'cuda:{}'.format(device))

        if len(batch.input_ids[0]) > 1024:
            if device == 'cpu':
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:]
            else:
                end_token = model.tokenizer.encode('</s>', return_tensors='pt')[0][-2:].to('cuda:{}'.format(device))


            input_ids = torch.cat((batch.input_ids[0][:1022], end_token), 0).unsqueeze(0)
            batch.input_ids = input_ids

        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            **generate_kwargs,
        )
        logprob = generate_id_to_logits(model,summaries,batch.input_ids,batch.attention_mask)
        dec = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i,hypothesis in enumerate(dec):
            output_list.append([hypothesis,logprob[i]])

    return output_list


def generate_summaries_or_translations_QA():
    return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.",
                        default="facebook/bart-large")
    parser.add_argument("--input_path", type=str, default="./data_train_BART/train.source", help="like cnn_dm/test.source")
    #parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default='', help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=1, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    ################################
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./pretrained/',
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    ################################
    parser.add_argument(
        "--ckpt_path",
        default="./models/NQ_then_FairytaleQA_Dev/epoch=1.ckpt",
        type=str,
        help='path tooo stored model checkpoints',
    )
    ################################
    parser.add_argument("--output_dir", default="./models/NQ_then_FairytaleQA_Dev/QAPairs/all/", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="facebook/bart-large",
                        help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--config_name", type=str, default="facebook/bart-large")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-large")
    parser.add_argument("--test_max_target_length", type=int,default=900)
    parser.add_argument("--eval_max_length", type=int,default=900)
    parser.add_argument("--type_embedding_enabled", type=bool, default=True)
    ################################
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    ################################
    args, rest = parser.parse_known_args()
    return args,rest

def set_env():
    if DEFAULT_DEVICE != 'cpu':
        predictor = Predictor.from_path(
            "./bert-base-srl-2020.11.19.tar.gz", cuda_device=0)
    else:
        predictor = Predictor.from_path(
            "./bert-base-srl-2020.11.19.tar.gz", )

    nlp = spacy.load("en_core_web_sm")
    return predictor,nlp

def set_model(args):
    model = SummarizationModule(args)
    model_rank = SummarizationModule(args)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()
    print("finish loading ACQ model")
    if DEFAULT_DEVICE != 'cpu':
        model.to('cuda:{}'.format(DEFAULT_DEVICE))
        model_rank.to('cuda:{}'.format(DEFAULT_DEVICE))
    return model


def answer_extraction_progress(input_content,nlp,predictor):
    # ==== ANSWER EXTRACTION PROCESS ====
    # extracted results are stored in a list, where each line is an A-C pair
    input_for_ACQ_model = []
    doc = nlp(input_content)
    json_samples = []
    # sent for sentences
    for sent in doc.sents:
        json_samples.append({'sentence': sent.text})
    srl_results = predictor.predict_batch_json(json_samples)

    each_answer = ''

    unique_answers = []
    # ent for entity
    for ent in doc.ents:
        each_answer = ent.text.replace('\n', '')
        each_answer = each_answer.strip('.,\'').strip() + ' .'
        # 这是为了避免产生重复的问答对
        if each_answer in unique_answers:
            continue
        if len(each_answer) > 1:
            input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
            unique_answers.append(each_answer)

    for chunk in doc.noun_chunks:
        each_answer = chunk.text.replace('\n', '')
        each_answer = each_answer.strip('.,\'').strip() + ' .'
        if each_answer in unique_answers:
            continue
        if len(chunk.text.split(" ")) >= 2:
            input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
            unique_answers.append(each_answer)

    for k in srl_results:
        if k['verbs'] == []:
            continue
        relevant_words = []
        for idx in range(len(k['verbs'][0]['tags'])):
            if k['verbs'][0]['tags'][idx] != '0':
                relevant_words.append(k['words'][idx])
        target = ' '.join(relevant_words)

        each_answer = target.replace("\n", "").strip('.,\'').strip() + ' .'

        if each_answer in unique_answers:
            continue

        input_for_ACQ_model.append(each_answer + " </s> " + input_content.replace("\n", ""))
        unique_answers.append(each_answer)
    # ==== END OF ANSWER EXTRACTION PROCESS ====
    return input_for_ACQ_model

def run_generate(examples,args,model):
    parsed = {}
    generate_results = generate_summaries_or_translations(
        model,
        examples,
        args.model_name_or_path,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        args=args,
        **parsed,
    )

    return generate_results

def e2e(input_content,input_category,nlp,predictor,args,model):
    print(" ==== ANSWER EXTRACTION PROCESS ====")
    input_for_ACQ_model = answer_extraction_progress(input_content,nlp,predictor)
    print("==== END OF ANSWER EXTRACTION PROCESS ====")

    print("==== QUESTION GENERATION PROCESS ====")
    output_for_ACQ_model = run_generate(input_for_ACQ_model,args,model)
    print(" ==== END OF QUESTION GENERATION PROCESS ====")

    # ==== POST-PROCESS ====
    for i in range(len(output_for_ACQ_model)):
        output_for_ACQ_model[i][0] = output_for_ACQ_model[i][0].replace("\n", "").split("?")[0].strip(' ?') + " ?"

    data_AC_Q = []

    for i in range(len(input_for_ACQ_model)):
        # data_AC_Q.append(input_for_ACQ_model[i].strip().lower() + " </s> " + output_for_ACQ_model[i].strip().lower())
        data_AC_Q.append([input_for_ACQ_model[i].strip().lower().split(' </s> ')[0],
                          input_for_ACQ_model[i].strip().lower().split(' </s> ')[1],
                          output_for_ACQ_model[i][0].strip().lower(), str(output_for_ACQ_model[i][1])])

    return data_AC_Q

def main():
    args,rest = get_args()
    predictor,nlp = set_env()
    model = set_model(args)
    prefix = 'val'
    print("The setting of enviroment is done!")
    book_directory = glob.glob('./data_generate_QA_pairs/'+prefix+'/*.txt',recursive=True)
    save_directory = glob.glob('./output_QA_pairs_lmscore/'+prefix+'/*.txt',recursive=True)
    saved_list = []
    for i in save_directory:
        book_name = i.split('/')[-1].replace('.txt', '').replace('generated_qa_stories-', '').lower()
        saved_list.append(book_name)

    cnt_book = 0
    for each_book in book_directory:
        book_name = each_book.split('/')[-1].replace('.txt', '').replace('QA_generation_stories-', '').lower()
        if book_name in saved_list:
            continue

        ACQ_pair_list = []
        print(book_name)
        lines = open(each_book,'r').readlines()
        for i in lines:
            ACQ_pair_list += e2e(i.replace('\n',''),'key',nlp,predictor,args,model)

        f_out = open('./output_QA_pairs_lmscore/'+prefix+'/generated_qa_stories-' + book_name + ".txt", 'w')

        for i in ACQ_pair_list:
            f_out.write(i[0] + ' </s> ' + i[1] + ' </s> ' + i[2] + ' </s> ' + i[3] +  '\n')
        f_out.close()

        print(cnt_book, book_name)

        cnt_book += 1

    print("book count:",cnt_book)


def generate_with_original_answer():
    args, rest = get_args()
    model = set_model(args)
    answers = []
    contexts = []
    input_for_ACQ_model = []
    with open(args.input_path,'r') as f:
        datas = f.readlines()
        f.close()
    for data in datas:
        data = data.replace('\n','').strip()
        answer = data.split('<SEP>')[0]
        context = data.split('<SEP>')[1]
        input_for_ACQ_model.append(answer + " </s> " + context)

    print("==== QUESTION GENERATION PROCESS ====")
    output_for_ACQ_model = run_generate(input_for_ACQ_model, args, model)
    print(" ==== END OF QUESTION GENERATION PROCESS ====")

    # ==== POST-PROCESS ====
    questions_scores = []
    for i in range(len(output_for_ACQ_model)):
        question = output_for_ACQ_model[i][0].replace("\n", "").split("?")[0].strip(' ?') + " ?"
        questions_scores.append((question.strip().lower(),str(output_for_ACQ_model[i][1])))

    with open(args.output_dir + 'train_generations.txt','w') as f:
        for item in questions_scores:
            f.write(item[0] + ' ' + item[1] + '\n')
        f.close()

def read_file(filename):
    with open(filename,'r') as f:
        datas = f.readlines()
    return datas


def filter(percentage = 1.0):
    generated_sources = read_file("./data_train_BART/train.source")
    generated_targets = read_file("./models/NQ_then_FairytaleQA_Dev/QAPairs/all/train_generations.txt")
    maps = {}
    scores = []
    sources = read_file("./models/NQ_then_FairytaleQA_Dev/QAPairs/ori_train.source")
    targets = read_file("./models/NQ_then_FairytaleQA_Dev/QAPairs/ori_train.target")
    #sources = []
    #targets = []
    for i,(source,target) in enumerate(zip(generated_sources,generated_targets)):
        source = source.replace('\n','').strip()
        target = target.replace('\n','').strip()
        question = target.split('? ')[0] + '?'
        lm_score = float(target.split('? ')[1])
        answer = source.split('<SEP>')[0]
        context = source.split('<SEP>')[1]
        maps[i] = (answer,question,context)
        scores.append((i,lm_score))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[:int(len(sorted_scores) * percentage)]
    for id,score in sorted_scores:
        sources.append(maps[id][1] + '<SEP>' + maps[id][2])
        targets.append(maps[id][0])

    print(f"sources:{len(sources)},targets:{len(targets)}.")

    with open("./models/NQ_then_FairytaleQA_Dev/QAPairs/all/train.source",'w') as f:
        for source in sources:
            f.write(source)
        f.close()
    with open("./models/NQ_then_FairytaleQA_Dev/QAPairs/all/train.target",'w') as f:
        for target in targets:
            f.write(target)
        f.close()


    return sources,targets



if __name__=="__main__":
    main()
    #generate_with_original_answer()
    #filter()