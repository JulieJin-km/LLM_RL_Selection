import openai
import argparse
import json
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch


from dataset import RLDataset
from Scorers import OPTScorer, FLANScorer, GPT3Scorer



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--QAG_file",
        default=None,
        type=str,
        help="The input training file.",
    )
    parser.add_argument("--output_dir",required=True)
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="The name of dataset. ",
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
    parser.add_argument(
        "--qa_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--score_type", default=None, type=str, help="the metric name of GPTScore")
    parser.add_argument("--api_key",default="",type=str,help='Valuable openai keys')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0 if not torch.cuda.is_available() else torch.cuda.device_count()

    return args

def get_logscore(Scorer,template, question, context, answer, prefix, asp_df, scorer_answer):
    inputs = template.replace('CCCCCC', context).replace('QQQQQQ', question).replace('AAAAAA',answer) + '\nAnswer:\n'

    # print(len(prefix + inputs))
    lens = len(prefix + inputs)

    prefix_input = prefix + inputs
    output = Scorer.score([prefix_input], [scorer_answer], prompt_text="\nAnswer: ")
    # score = np.array(output)
    score = output[0]
    return score


def main():
    args = get_argument()
    print(args)
    set_seed(args)
    if args.dataset_name == "squad" or args.dataset_name == "NaturalQuestionsShort" or args.dataset_name == "TriviaQA-web":
        data_type = "json"
        dataset_type = "ext"
    else:
        data_type = "txt"
        dataset_type = "gen"

    dataset_rl = RLDataset(args.QAG_file, args.dataset_name, data_type, tokenizer=None,
                           max_seq_length=args.max_seq_length, doc_stride=args.doc_stride,
                           max_query_length=args.max_query_length, overwrite_cache=True, need_features=False)

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
    }
    checkpoint = args.qa_model_name_or_path
    if args.score_type in ["flan_small_score", "flan_base_score", "flan_large_score", "flan_xl_score",
                           "flan_xxl_score"]:
        Scorer = FLANScorer(device=args.device, checkpoint=checkpoint)
    elif args.score_type in [
        "opt125m_score", "opt350m_score", "opt1_3B_score",
        "opt2_7B_score", "opt6_7B_score", "opt13B_score", "opt30B_score", "opt66B_score",
        "gpt2_medium_score", "gpt2_large_score", "gpt2_xl_score", "gptJ6B_score",'llama2_score','llama2_chat_score'
    ]:
        Scorer = OPTScorer(device=args.device, checkpoint=checkpoint)
    else:
        Scorer = GPT3Scorer(model_name=checkpoint, api_key=args.api_key)


    asp_data = json.load(open("./data/llm_instruction_design.json"))
    asp_dfs = asp_data["asp_definition"]
    asp_df_d = asp_dfs['SEM'].strip()
    asp_df_m = asp_dfs['MRC'].strip()
    asp_df_q = asp_dfs['QUE'].strip()
    # print('demos: ',self.demos)
    # print('asp_df:', asp_df)
    dialog_template = ["Conversation: Human asked, CCCCCC. QQQQQQ AI answered AAAAAA",
                       "Conversation: Human: CCCCCCC. QQQQQQ AI: AAAAAA",
                       "Conversation: Human asked, CCCCCC. Based on the context, QQQQQQ AI answered AAAAAA",
                       "Conversation: Human: CCCCCCC. Based on the context, QQQQQQ AI: AAAAAA",
                       "Conversation: Human said, answer the question based on the given text. The question is QQQQQQ The text is CCCCCC. AI answered AAAAAA"]
    mrc_template = ["Question: QQQQQQ\nContext: CCCCCC"]
    question_template = [
        "Conversation: Human said, generate a question based on the following text. CCCCCC. AI said QQQQQQ"]
    m_template = mrc_template[0]
    prompt_type = 0
    d_template = dialog_template[prompt_type]
    q_template = question_template[0]
    prefix_d = asp_df_d + '\n'
    prefix_m = asp_df_m + '\n'
    prefix_q = asp_df_q + '\n'

    score_type = 0
    writethrough = True
    if args.score_type == "gpt3_score":
        try:
            datas = json.load(open(os.path.join(args.output_dir, args.qa_model_name_or_path + '_' + str(score_type) + '.json'),'r'))

        except:
            datas = {}
        filename = args.qa_model_name_or_path + '_score_' + str(score_type) + '_prompt_' + str(prompt_type) + '.json'
        only_all = False
    else:
        filename = args.score_type + '_' + str(score_type) + '_prompt_' + str(prompt_type) + '.json'
        try:
            datas = json.load(open(os.path.join(args.output_dir, args.score_type + '_score_' + str(score_type) + '_prompt_' + str(prompt_type) + '.json'),'r'))
            print(len(datas))
        except:
            datas = {}
        only_all = False
    print(len(datas))
    for i,example in enumerate(dataset_rl.examples):
        qid = example.qas_id
        if qid in datas:
            continue
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(dataset_rl.examples)}")
            if writethrough:
                with open(os.path.join(args.output_dir, filename),'w') as f:
                    json.dump(datas, f, indent=4)
                    f.close()

        context = example.context_text
        question = example.question_text
        if question[-1] != '?':
            question = question + '?'
        answer = example.answer_text + '.'
        if answer[-1] != '.':
            answer = answer + '.'
        contexts = []
        all_context = context
        contexts.append(all_context)
        if not only_all:
            window_size = 1000
            doc_stride = 512
            i = 0
            if i + window_size < len(all_context):
                while i < len(all_context):
                    sub_context = context[i:i + window_size]
                    contexts.append(sub_context)
                    if i + window_size >= len(all_context):
                        break
                    i += doc_stride
        print(len(contexts))
        sub_scores = []
        for context in contexts:
            if score_type == 0:
                score_d = get_logscore(Scorer,d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                score = score_d
            elif score_type == 1:
                score_d = get_logscore(Scorer,d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                score_m = get_logscore(Scorer,m_template, question, context, answer, prefix_m, asp_df_m, answer)
                score = (score_d + score_m) / 2
            elif score_type == 2:
                score_q = get_logscore(Scorer,q_template, question, context, answer, prefix_q, asp_df_q, "Yes")
                score_d = get_logscore(Scorer,d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                score = (score_q + score_d) / 2
            elif score_type == 3:
                score_q = get_logscore(Scorer,q_template, question, context, answer, prefix_q, asp_df_q, "Yes")
                score_d = get_logscore(Scorer,d_template, question, context, answer, prefix_d, asp_df_d, "Yes")
                score_m = get_logscore(Scorer,m_template, question, context, answer, prefix_m, asp_df_m, answer)
                score = (score_q + score_d + score_m) / 3


            sub_scores.append(score)
            if score == -15.0:
                break
        datas[qid] = max(sub_scores)

    with open(os.path.join(args.output_dir, filename),'w') as f:
        json.dump(datas, f, indent=4)
        f.close()



if __name__ == '__main__':
    main()