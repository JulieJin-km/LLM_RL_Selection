import torch
import torch.nn as nn
import traceback
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, GPTJForCausalLM, AutoModelForCausalLM, AutoTokenizer
import openai
import time
import sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



class FLANScorer:
    def __init__(self, device='cuda:0', max_length=5000, checkpoint='google/flan-t5-base'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load('models/bart.pth', map_location=self.device))

    def score(self, srcs, tgts, prompt_text = None,batch_size = 1):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            #if i <1:
                #print('src_list: ',src_list)
                #print('tgt_list: ', tgt_list)
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

class OPTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint=None):
        # Set up model
        self.device = device

        if 'gpt2' in checkpoint:
            print('gpt2 model')
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint).to(self.device)
            max_length = 1000
            #max_length = 2000
        elif 'gpt-j' in checkpoint:
            print('gpt-j model')
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
            self.model = GPTJForCausalLM.from_pretrained(checkpoint).to(self.device)
            max_length = 2000
        elif 'llama' in checkpoint:
            print(checkpoint)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=True, trust_remote_code=True,use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map='auto',
                                                         use_auth_token=True, trust_remote_code=True)
            max_length = 4000
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
            self.model = OPTForCausalLM.from_pretrained(checkpoint).to(self.device)
            max_length = 2000
        self.max_length = max_length
        print('max_length: ', max_length)
        self.model.eval()

    def score(self, srcs, tgts, prompt_text, batch_size=1):
        """ Score a batch of examples """

        def trunk_input(inputs, outputs, reduce_seq, max_length):
            input_ids = self.tokenizer.encode(inputs)[1:-1]
            output_ids = self.tokenizer.encode(outputs)[1:-1]
            reduce_seq_ids = self.tokenizer.encode(reduce_seq)[1:-1]
            total_len = len(input_ids) + len(output_ids)
            if total_len > max_length:
                del_len = len(input_ids) + len(output_ids) - max_length
                reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
                reduce_seq = self.tokenizer.decode(reduce_seq_ids[1:-1])
            return reduce_seq

        score_list = []
        for i, (src, tgt) in enumerate(zip(srcs, tgts)):
            #print('process:' + str(i) + '/' + str(len(srcs)))
            new_src = trunk_input(src, tgt, src, max_length=self.max_length)
            src = new_src
            text = src + tgt
            #if i < 1:
                #print('text: ', text)
                #print('tgt: ', tgt)
            input_ids = self.tokenizer.encode(text)
            tgt_ids = self.tokenizer.encode(tgt)
            output_ids = [-100] * len(input_ids)
            output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=output_ids,
                )
            loss, logits = outputs[0], outputs[1]
            loss = loss.item()
            score = -loss
            score_list.append(score)
            '''
            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=output_ids,
                    )
                loss, logits = outputs[0], outputs[1]
                loss = loss.item()
                score = -loss
                score_list.append(score)
                #print('score: ', score)
            except RuntimeError:
                # traceback.print_exc()
                print('input_ids: ', input_ids)
                print('output_ids: ', output_ids)
                print(f'source: {src}')
                print(f'target: {tgt}')
                # exit(0)
            '''
        return score_list

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


class GPT3Scorer(object):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except:
            print("Something wrong with api_keys")
            exit(-1)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                response = completion_with_backoff(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                #print('prompt: ',prompt)
                received = True
                #time.sleep(3)
            except:
                error = sys.exc_info()[0]
                print(sys.exc_info())
                print(prompt)
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
                return None
        return response

    def do_inference(self, input, output, max_length = 2048):
        losses = []
        data = input + output

        response = self.gpt3(data)
        if response is None:
            return 15.0
        out = response["choices"][0]

        assert input + output == out["text"]
        i = 0
        # find the end position of the input...
        i = out['logprobs']['text_offset'].index(len(input) - 1)
        if i == 0:
            i = i + 1
        #print('eval text', out['logprobs']['tokens'][i + 1: ])
        loss = -sum(out['logprobs']["token_logprobs"][i + 1:])  # ignore the last '.'
        avg_loss = loss / (len(out['logprobs']['text_offset']) - i - 1)  # 1 is the last '.'
        print('avg_loss: ', avg_loss)
        losses.append(avg_loss)

        return avg_loss

    def score(self,inputs,outputs,prompt_text):
        score_list = []
        for input,output in zip(inputs,outputs):
            avg_loss = self.do_inference(input,output)
            score = -avg_loss
            score_list.append(score)
        return score_list