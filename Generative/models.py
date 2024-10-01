
from lightning_base import BaseTransformer,add_generic_args
from utils import use_task_specific_params,freeze_params,assert_all_frozen,lmap
from utils import Seq2SeqDataset,LegacySeq2SeqDataset
from utils import label_smoothed_nll_loss,flatten_list,save_json,calculate_rouge,calculate_bleu



import argparse
from argparse import Namespace
from pathlib import Path
from collections import defaultdict
from transformers import MBartTokenizer,T5ForConditionalGeneration
from typing import List,Tuple,Dict
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import numpy as np
import time
from torch.utils.data import DataLoader


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
class SummarizationModule(BaseTransformer):
    mode = 'summarization' # 说明任务名称
    loss_names = ['loss']
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"
    istest = True

    def __init__(self,hparams,**kwargs):
        if self.istest:
            super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
            # self.mode是类共用的，self.model是Namespace里的
            use_task_specific_params(self.model, self.mode)
            return

        ############################
        if 'sortish_sampler' not in hparams:
            hparams['sortish_sampler'] = False
            hparams = Namespace(**hparams)
        if 'num_workers' not in hparams:
            hparams['num_workers'] = 0
            hparams = Namespace(**hparams)
        if type(hparams) != argparse.Namespace:
            hparams = Namespace(**hparams)
        ############################
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        # self.mode是类共用的，self.model是Namespace里的
        use_task_specific_params(self.model, self.mode)
        #         save_git_info(self.hparams.output_dir)
        # self.output_dir也是Namespace里的
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        #         pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        # ========= TIMER ==========
        self.timer_count = 0
        self.timer_sum = 0

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
            #######################
            # fuse_num=self.hparams.fuse_num,
            # type_embedding=self.hparams.type_embedding,
            #######################
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        # hhh可以玩一玩
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = 2  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            # The MBart model was presented in Multilingual Denoising Pre-training for Neural Machine Translation ,
            # According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on
            # large-scale monolingual corpora in many languages using the BART objective.
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            Seq2SeqDataset if hasattr(self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )

        ##################
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}.Need an integer > 1"
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder,self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder,self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        output = self.model(input_ids, **kwargs)
        return output
        '''
        try:
            output = self.model(input_ids,**kwargs)
            return output
        except:
            print("ERROR:",input_ids.shape)
        '''

    def ids_to_clean_text(self,generated_ids:List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def generate(self,input_ids,attention_mask,**generate_kwargs):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=20,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=3,
            early_stopping=True,
            use_cache=False,
            **generate_kwargs
        )
        return generated_ids

    def _step(self,batch:dict) -> Tuple:
        pad_token_id =self.tokenizer.pad_token_id
        src_ids,src_mask = batch["input_ids"],batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model,T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            #decoder_input_ids = shift_tokens_right(tgt_ids,pad_token_id,self.decoder_start_token_id)
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, 0)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                        #freeze_encoder=self.hparams.freeze_encoder,
                        #event_ids=batch['event_ids'] if 'event_ids' in batch else None
                       )
        try:
            lm_logits = outputs[0]
        except:
            print("==== ERROR IN _step: ====")
            print("input_ids: ", batch["input_ids"].shape)
            print("attention_mask: ", batch["attention_mask"].shape)
            print("labels: ", batch["labels"].shape)
            print("outputs: ", outputs.shape)
            print("==== END ERROR PRINT ====")

        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1,lm_logits.shape[-1]),tgt_ids.view(-1))

        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits,dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)


    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name:loss for name,loss in zip(self.loss_names,loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        return {"loss":loss_tensors[0],"log":logs}

    def validation_step(self, batch,batch_idx) -> Dict:
        return self._generative_step(batch)

    def save_metrics(self,latest_metrics,type_path):
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics,self.metrics_save_path)

    def validation_epoch_end(self, outputs,prefix="val") -> Dict:
        self.step_count += 1
        losses = {k:torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k:v.item() for k,v in losses.items()})
        losses.update(generative_metrics)
        # dict的key可以这么造呀
        all_metrics = {f"{prefix}_avg_{k}":x for k,x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.save_metrics(all_metrics,prefix)
        preds = flatten_list([x["preds"] for x in outputs])

        return {
            "log":all_metrics,
            "preds":preds,
            f"{prefix}_loss":loss,
            f"{prefix}_{self.val_metric}":metric_tensor
        }

    def calc_generative_metrics(self,preds,target) -> Dict:
        return calculate_rouge(preds,target)

    # 这应该就是关键的训练函数
    def _generative_step(self,batch:dict) -> dict:
        t0 = time.time()
        ########################################
        generate_kwargs = dict()
        # generate_kwargs['fuse_num'] = self.hparams.fuse_num
        # generate_kwargs['type_embedding'] = self.hparams.type_embedding
        ########################################
        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
            ###################
            **generate_kwargs,
            ###################
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        # print("generated input: ", batch["input_ids"])
        # print("generated output: ", generated_ids)
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self,batch,batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs,prefix="test")

    def get_dataset(self,type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset=self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            sampler = dataset.make_sortish_sampler(batch_size,distributed=self.hparams.gpus > 1)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train",batch_size=self.hparams.train_batch_size,shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("val",batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test",batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser,root_dir)
        add_generic_args(parser,root_dir)
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")

        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        # n_obs
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        # task
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )

        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)

        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)

        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "rouge2", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")

        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        ############################
        parser.add_argument(
            "--ckpt_path",
            default=None,
            type=str,
            help='path tooo stored model checkpoints',
        )
        parser.add_argument(
            "--fuse_num",
            default=None,
            type=int,
            help='num of passage vector to fuse in decoder',
        )
        parser.add_argument(
            "--type_embedding",
            action="store_true",
            help='whether to add a type embedding layer during encoding',
        )
        ############################
        return parser

class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self,hparams,**kwargs):
        super().__init__(hparams,**kwargs)
        self.dataset_kwargs['src_lang'] = hparams.src_lang
        self.dataset_kwargs['tgt_lang'] = hparams.tgt_lang

    def calc_generative_metrics(self,preds,target):
        return calculate_bleu(preds,target)