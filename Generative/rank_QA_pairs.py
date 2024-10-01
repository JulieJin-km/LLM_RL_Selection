from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from transformers import TrainingArguments,Trainer
import torch
import glob
import random
import json
import os

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    training_args = TrainingArguments(
        output_dir='./rank_result',
        learning_rate=1.541663609697611e-05,
        num_train_epochs=4,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./rank_result',
        logging_steps=30000,
        save_steps=100000,
        seed=15
    )
    # lr这个神奇的数字是从哪里来的
    model = DistilBertForSequenceClassification.from_pretrained(
        "./RANKING_QA_DistilBert_QAC_lr1e-5_e4_seed15_b1_uneven_distributed", local_files_only=True
    )
    trainer = Trainer(
        model=model,
        args=training_args
    )

    prefix = 'val'
    N = 20
    save_directory = './rank_top'+str(N)+'_result/'+prefix
    generation_results = []
    total_generated_results = 0
    cnt_book = 0
    book_directory = glob.glob("./output_QA_pairs/"+prefix+"/*.txt", recursive=True)

    for each_book in book_directory:
        key = each_book.split('/')[-1].replace('.txt', '').replace('generated_qa_stories-', '').lower()
        generation_answers = []
        generation_data_for_ranking = []
        lines_generate = open(each_book,'r').readlines()
        section_idx = {}
        section_count = {}

        for idx,i in enumerate(lines_generate):
            whole = i.replace('\n','')
            generation_answers.append(whole)
            question = whole.split('</s>')[-1].strip()
            answer = whole.split('</s>')[0].strip()
            section = whole.split('</s>')[1].strip()
            section_idx[idx] = section
            section_count[section] = 0

            # Q + A +C
            generation_data_for_ranking.append(question + ' <SEP> ' + answer + ' <SEP> ' + section)

            total_generated_results += 1

        # create the evaluation dataset for the ranking model, and run through the model
        test_encodings = tokenizer(generation_data_for_ranking, truncation=True, padding=True)
        # 注意label的生成方式
        test_labels = [1] * len(generation_data_for_ranking)
        test_dataset = RankingDataset(test_encodings, test_labels)

        predictions = trainer.predict(test_dataset)
        assert(len(predictions.predictions)==len(generation_answers))

        # Here you may store all ranked QA-pairs along with the score into a file
        # Or just directly sort by the values and select the top-N ones

        # store all
        '''
        f1 = open(save_directory + '/ranked_generated_qa_stories-' + key + ".txt", 'w')

        # 看一下predictions的形式
        for i in range(len(predictions.predictions)):
            predicted = predictions.predictions[i][1]
            f1.write(generation_answers[i] + '<SEP>' + str(predicted) + '\n')

        f1.close()
        '''
        # top N per section
        qa_pairs = [(generation_answers[i],predictions.predictions[i][1],i) for i in range(len(predictions.predictions))]
        qa_pairs.sort(key=lambda item:item[1])
        #qa_pairs.reverse()

        f1 = open(save_directory + '/ranked_generated_qa_stories-' + key + ".txt", 'w')
        for qa_pair in qa_pairs:
            sec = section_idx[qa_pair[2]]
            section_count[sec] += 1
            if section_count[sec] > N:
                continue
            f1.write(qa_pair[0] + '<SEP>' + str(qa_pair[1]) + '\n')

        f1.close()
        cnt_book += 1


def random_sample():
    prefix = 'val'
    N = 20
    seed = 42
    random.seed(seed)
    save_directory = './random_'+ str(seed) + '_' + str(N) + '_result/' + prefix
    generation_results = []
    total_generated_results = 0
    cnt_book = 0
    book_directory = glob.glob("./output_QA_pairs/" + prefix + "/*.txt", recursive=True)


    for each_book in book_directory:
        key = each_book.split('/')[-1].replace('.txt', '').replace('generated_qa_stories-', '').lower()
        generation_answers = []
        lines_generate = open(each_book, 'r').readlines()
        section_idx = {}
        section_count = {}

        for idx, i in enumerate(lines_generate):
            whole = i.replace('\n', '')
            generation_answers.append(whole)
            section = whole.split('</s>')[1].strip()
            section_idx[idx] = section
            section_count[section] = 0

            total_generated_results += 1

        qa_pairs = [(generation_answers[i], i) for i in range(len(generation_answers))]
        random.shuffle(qa_pairs)

        f1 = open(save_directory + '/random_generated_qa_stories-' + key + ".txt", 'w')
        for qa_pair in qa_pairs:
            sec = section_idx[qa_pair[1]]
            section_count[sec] += 1
            if section_count[sec] > N:
                continue
            f1.write(qa_pair[0] + '<SEP>' + str(0.0) + '\n')


        f1.close()
        cnt_book += 1

def lm_score():
    prefix = 'val'
    N = 20
    save_directory = './lmscore_' + str(N) + '_result/reverse_' + prefix
    generation_results = []
    total_generated_results = 0
    cnt_book = 0
    book_directory = glob.glob("./output_QA_pairs_lmscore/" + prefix + "/*.txt", recursive=True)

    for each_book in book_directory:
        key = each_book.split('/')[-1].replace('.txt', '').replace('generated_qa_stories-', '').lower()
        generation_answers = []
        lines_generate = open(each_book, 'r').readlines()
        section_idx = {}
        section_count = {}
        lm_scores = {}

        for idx, i in enumerate(lines_generate):
            whole = i.replace('\n', '')
            generation_answers.append(whole)
            section = whole.split('</s>')[1].strip()
            score = float(whole.split('</s>')[-1].strip())
            lm_scores[idx] = score
            section_idx[idx] = section
            section_count[section] = 0

            total_generated_results += 1

        qa_pairs = [(generation_answers[i],i, lm_scores[i]) for i in range(len(generation_answers))]
        sorted(qa_pairs,key=lambda  x:x[2],reverse=False)


        f1 = open(save_directory + '/lmscore_generated_qa_stories-' + key + ".txt", 'w')
        for qa_pair in qa_pairs:
            sec = section_idx[qa_pair[1]]
            section_count[sec] += 1
            if section_count[sec] > N:
                continue
            f1.write(qa_pair[0] + '<SEP>' + str(0.0) + '\n')

        f1.close()
        cnt_book += 1


def competency():
    prefix = 'val'
    N = 20
    type = "main_all"
    save_directory = './competency_'  + str(N) + '_result/' + type
    os.makedirs(save_directory, exist_ok=True)
    with open('./output_QA_pairs/' + prefix + '/keyed_difficulty/' + type + '.json','r') as f:
        scores = json.load(f)
        f.close()
    total_generated_results = 0
    cnt_book = 0
    book_directory = glob.glob("./output_QA_pairs/" + prefix + "/*.txt", recursive=True)
    book_directory = sorted(book_directory)

    for book_idx,each_book in enumerate(book_directory):
        key = each_book.split('/')[-1].replace('.txt', '').replace('generated_qa_stories-', '').lower()
        generation_answers = []
        lines_generate = open(each_book, 'r').readlines()
        section_idx = {}
        section_count = {}
        sec_scores = {}

        for line_idx, i in enumerate(lines_generate):
            whole = i.replace('\n', '')
            if len(whole) == 0:
                continue
            generation_answers.append(whole)
            section = whole.split('</s>')[1].strip()
            qid = str(book_idx) + '_' + str(line_idx)
            score = float(scores[qid])
            sec_scores[line_idx] = score
            section_idx[line_idx] = section
            section_count[section] = 0

            total_generated_results += 1

        qa_pairs = [(generation_answers[i],i, sec_scores[i]) for i in range(len(generation_answers))]
        sorted(qa_pairs,key=lambda  x:x[2],reverse=False)


        f1 = open(save_directory + '/' + type + '_generated_qa_stories-' + key + ".txt", 'w')
        for qa_pair in qa_pairs:
            sec = section_idx[qa_pair[1]]
            section_count[sec] += 1
            if section_count[sec] > N:
                continue
            f1.write(qa_pair[0] + '<SEP>' + str(0.0) + '\n')

        f1.close()
        cnt_book += 1

if __name__=="__main__":
    #main()
    random_sample()
    #lm_score()
    #competency()









