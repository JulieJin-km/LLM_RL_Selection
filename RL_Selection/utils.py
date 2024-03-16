# No need to modify when changing downstream datasets
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


from logging import getLogger
from torch import nn
from typing import List,Iterable,Callable,Dict
from torch.utils.data import Dataset,Sampler
from pathlib import Path
from transformers import BartTokenizer
import numpy as np
import torch.distributed as dist
import math
from transformers.file_utils import cached_property
import torch
import linecache
import json
import itertools
from rouge_score import rouge_scorer,scoring
from sacrebleu import corpus_bleu


logger = getLogger(__name__)
def use_task_specific_params(model,task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task,{})
        logger.info(f"using task specific params for {task}:{pars}")
        model.config.update(pars)

def freeze_params(model:nn.Module):
    for par in model.parameters():
        par.requires_grad = False

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def assert_all_frozen(model):
    #List、列表，是 list 的泛型，基本等同于 list，其后紧跟一个方括号，里面代表了构成这个列表的元素类型：
    model_grads:List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int,model_grads))
    npars = len(model_grads)
    # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

def sortish_sampler_indices(data,bs):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    def key_fn(i):
        return data[i]
    # here is a bit of randomness
    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i:i+sz] for i in range(0,len(idxs),sz)]
    sort_idx = np.concatenate([sorted(s,key=key_fn,reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i:i+sz] for i in range(0,len(sort_idx),sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx)>1 else np.array([],dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0],sort_idx))
    return sort_idx


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    def __init__(self,data,batch_size):
        self.data,self.bs = data,batch_size

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs))

class DistributedSortishSampler(Sampler):
    def __init__(self,dataset,batch_size,num_replicas=None,rank=None,add_extra_example=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_example:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_example

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data,self.batch_size)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank:self.total_size:self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self,epoch):
        self.epoch = epoch

class AbstractSeq2SeqDataset(Dataset):
    def __init__(self,tokenizer,data_dir,max_source_length,max_target_length,type_path="train",n_obs=None,
                 src_lang=None,tgt_lang=None,prefix=""):
        super().__init__()
        # 超级重要
        self.src_file = Path(data_dir).joinpath(type_path+".source")
        self.tgt_file = Path(data_dir).joinpath(type_path+".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""
        # 用于截取
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.add_prefix_space = isinstance(self.tokenizer,BartTokenizer)

    def __len__(self):
        return len(self.src_lens)


    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def make_sortish_sampler(self,batch_size,distributed=False,**kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size)

    # 用于子类的继承
    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self,batch):
        raise NotImplementedError("You must implement this")

def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
    """Only used by LegacyDataset"""
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file),index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line,f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer,source_line,self.max_source_length)
        target_inputs = encode_line(self.tokenizer,tgt_line,self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return{
            "input_ids":source_ids,
            "attention_mask":src_mask,
            "labels":target_ids
        }

    def collate_fn(self,batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch

class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __getitem__(self, index) -> Dict[str,str]:
        index = index + 1 # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file),index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file),index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts":tgt_line,"src_texts":source_line,"id":index-1}

    def collate_fn(self,batch):
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            #add_prefix_space=self.add_prefix_space,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding

def label_smoothed_nll_loss(lprobs,target,epsilon,ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def save_json(content,path):
    with open(path,"w") as f:
        json.dump(content,f,indent=4)

def flatten_list(summary_ids:List[list]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
def calculate_rouge(output_lns:List[str],reference_lns:List[str],use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS,use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln,output_ln in zip(reference_lns,output_lns):
        scores = scorer.score(reference_ln,output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k:round(v.mid.fmeasure*100,4) for k,v in result.items()}

def calculate_bleu(output_lns,reference_lns,**kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [reference_lns], **kwargs).score, 4)}


if __name__ == "__main__":
    output_lns = ["I did it"]
    reference_lns = ["I did it as well"]
    output = calculate_rouge(output_lns,reference_lns)
    print(output)
