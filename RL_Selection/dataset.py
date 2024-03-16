# For RLAgent: input is QA pairs
# For reward, input is filtered_QA pairs(formatted)
import json
import logging
import numpy as np
import os
import torch
import collections
from torch.utils.data import TensorDataset
from multiprocessing import Pool, cpu_count
from functools import partial
# 简单说就是把一个函数,和该函数所需传的参数封装到一个class 'functools.partial'的类中,简化以后的调用方式
from tqdm import tqdm
from transformers.file_utils import is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.data.processors.squad import SquadV1Processor,SquadV2Processor,squad_convert_example_to_features,squad_convert_examples_to_features,squad_convert_example_to_features_init as tokenizer_global
import random

from utils import _is_whitespace,MULTI_SEP_TOKENS_TOKENIZERS_SET,_new_check_is_max_context

logger = logging.getLogger(__name__)

def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def CQA_convert_example_to_features(
        example, max_seq_length, doc_stride, max_query_length, padding_strategy
):
    features = []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    spans = []


    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # all_doc_tokens=all_doc_tokens[:tok_start_position] + ["[unused100]"] + all_doc_tokens[tok_start_position:tok_end_position+1] + ["[unused101]"] + all_doc_tokens[tok_end_position+1:]

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0
        p_mask[cls_index] = 0

        features.append(
            CQAFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                qas_id=example.qas_id,
                answer_text=example.answer_text
            )
        )
    return features



def CQA_encapsulation_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        padding_strategy="max_length",
        threads = 1,
        tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
    Returns:
        list of :class: CQAFeatures
    """

    # Defining helper methods
    features = []
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            CQA_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert RL examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    # return pt dataset
    if not is_torch_available():
         raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

    # prepare input_ids for RL model
    new_features = []
    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    all_rl_input_ids = []
    all_rl_attention_masks = []
    all_rl_token_type_ids = []
    special_tokens_ids = tokenizer.additional_special_tokens_ids

    for ex, efs in example_index_to_features.items():
        tmp_feature = efs[0]  # cls

        ######################################
        question_len = tmp_feature.input_ids.index(tokenizer.sep_token_id) + 1
        new_para_ids = tmp_feature.tokens[question_len:-1]

        new_question_ids = tmp_feature.input_ids[1:question_len - 1]

        truncated_answer = tokenizer.encode(tmp_feature.answer_text, add_special_tokens=False, truncation = True, max_length=max_query_length)

        #new_question_ids = new_question_ids + [special_tokens_ids[0]] + [special_tokens_ids[1]]
        new_question_ids = new_question_ids + [special_tokens_ids[0]] + truncated_answer

        encoded_dict = tokenizer.encode_plus(
            new_question_ids,
            new_para_ids,
            truncation=True,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_token_type_ids=True,
        )

        rl_input_id = encoded_dict.data['input_ids']
        rl_attention_mask = encoded_dict.data['attention_mask']
        rl_token_type_id = encoded_dict.data['token_type_ids']

        all_rl_input_ids.append(rl_input_id)
        all_rl_attention_masks.append(rl_attention_mask)
        all_rl_token_type_ids.append(rl_token_type_id)

        all_input_ids.append(tmp_feature.input_ids)
        all_attention_masks.append(tmp_feature.attention_mask)
        all_token_type_ids.append(tmp_feature.token_type_ids)

        new_features.append(tmp_feature)

    all_rl_input_ids = torch.tensor(all_rl_input_ids, dtype=torch.long)
    all_rl_attention_masks = torch.tensor(all_rl_attention_masks, dtype=torch.long)
    all_rl_token_type_ids = torch.tensor(all_rl_token_type_ids, dtype=torch.long)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    dataset = TensorDataset(
        all_rl_input_ids,
        all_rl_attention_masks,
        all_rl_token_type_ids,
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_feature_index,
    )
    features = new_features
    return features, dataset

def squad_encapsulation_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    # filter based on the original version
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=tokenizer_global, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset

        if not is_training:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            example_index_to_features = collections.defaultdict(list)
            for feature in features:
                example_index_to_features[feature.example_index].append(feature)

            all_input_ids = []
            all_attention_masks = []
            all_token_type_ids = []
            all_start_positions = []
            all_end_positions = []
            new_features=[]

            for ex, efs in example_index_to_features.items():
                tmp_feature = None
                for ef in efs:
                    if ef.start_position > 0 and ef.end_position > 0:
                        if tmp_feature:
                            if ef.start_position < tmp_feature.start_position:
                                tmp_feature = ef
                        else:
                            tmp_feature = ef

                if not tmp_feature:
                    tmp_feature = efs[0]
                # filter
                if tmp_feature.start_position==0 and tmp_feature.end_position==0:
                    continue
                all_input_ids.append(tmp_feature.input_ids)
                all_attention_masks.append(tmp_feature.attention_mask)
                all_token_type_ids.append(tmp_feature.token_type_ids)
                all_start_positions.append(tmp_feature.start_position)
                all_end_positions.append(tmp_feature.end_position)
                new_features.append(tmp_feature)

            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
            all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
            all_start_positions = torch.tensor(all_start_positions, dtype=torch.long)
            all_end_positions = torch.tensor(all_end_positions, dtype=torch.long)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_feature_index
            )
            features = new_features

        return features, dataset
    else:
        return features

class CQAFeatures:
    """
        Single example features to be fed to a model.
        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token indices.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs.
            cls_index: the index of the CLS token.
            p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
                Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
            example_index: the index of the example
            unique_id: The unique Feature identifier
            paragraph_len: The length of the context
            token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
                If a token does not have their maximum context in this feature object, it means that another feature object
                has more information related to that token and should be prioritized over this feature for that token.
            tokens: list of tokens corresponding to the input ids
            token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            qas_id: str = None,
            answer_text = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.qas_id = qas_id
        self.answer_text = answer_text

class CQAExample:
    """
        A single training/test example for the MRC dataset(both extractive and generative), as loaded from disk.
        All questions have corresponding answer.

        Args:
            qas_id: The example's unique identifier
            question_text: The question string
            context_text: The context string
            answer_text: The answer string
            answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            answers=[],
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.answers = answers


        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset



class RLDataset(object):
    def __init__(self,filename,dataset,datatype,tokenizer,max_seq_length = 384,doc_stride = 128,
                 max_query_length = 128,overwrite_cache = True,need_features = True,percentage=1.0):
        self.filename = filename
        self.dataset_name = dataset # the name of dataset
        self.type = datatype # json or text
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.overwrite_cache = overwrite_cache
        self.need_features = need_features
        self.features, self.dataset, self.examples = self.convert_to_examples_features(percentage)



    def get_datas(self):
        examples = []
        with open(self.filename, 'r') as f:
            datas = json.load(f)
            f.close()
        if self.type == "json":
            for passage in datas["data"]:
                for para in passage["paragraphs"]:
                    context = para["context"]
                    for qa in para['qas']:
                        question = qa["question"]
                        answer = qa["answers"][0]["text"]
                        qid = qa["id"]
                        example = CQAExample(
                            qas_id = qid,
                            question_text=question,
                            context_text=context,
                            answer_text=answer,
                            answers=qa["answers"]
                        )
                        examples.append(example)
            logger.info("Reading {} QA pairs from {}".format(len(examples),self.filename))
            return examples
        elif self.type == "txt":
            for data in datas['data']:
                example = CQAExample(
                    qas_id=data["qid"],
                    question_text=data["question"],
                    context_text=data["context"],
                    answer_text=data["answer"],
                    answers=[data["answer"]]
                )
                examples.append(example)

            logger.info("Reading {} QA pairs from {}".format(len(examples),self.filename))
            return examples
        return examples

    def convert_to_examples_features(self,percentage):
        if percentage == 1.0:
            cached_features_file = os.path.join(
                "cached_rl_{}_{}_{}".format(
                    self.dataset_name,
                    str(self.max_seq_length),
                    str(self.max_query_length)
                ),
            )
        else:
            cached_features_file = os.path.join(
                "cached_rl_{}_{}_{}_{}".format(
                    self.dataset_name,
                    str(self.max_seq_length),
                    str(self.max_query_length),
                    str(percentage)
                ),
            )
        if os.path.exists(cached_features_file) and not self.overwrite_cache:
            logger.info("Loading RL features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            return (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            logger.info("Creating RL features from dataset file at: %s", self.filename)
            examples = self.get_datas()
            if percentage != 1.0:
                examples = random.sample(examples,int(len(examples) * percentage))
                logger.info("Random sample %d examples", len(examples))

            if not self.need_features:
                logger.info("Only load %d examples, no feature extracting.",len(examples))
                return [],[],examples
            logger.info("Start extracting RL features")
            features, dataset = CQA_encapsulation_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
            return features,dataset,examples


class SquadDataset(object):
    def __init__(self,filename,tokenizer,version,istrain,max_seq_length=384,doc_stride=128,max_query_length=64,overwrite_cache=True):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.overwrite_cache = overwrite_cache
        if version == "1.0":
            self.processcor = SquadV1Processor()
        else:
            self.processcor = SquadV2Processor()
        cached_features_file = os.path.join(
            "cached_qa_{}_{}_{}".format(
                self.filename.split('/')[-1],
                str(self.max_seq_length),
                str(self.max_query_length)
            ),
        )
        if os.path.exists(cached_features_file) and not self.overwrite_cache:
            logger.info("Loading qa features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            self.examples = features_and_dataset["examples"]
            self.features = features_and_dataset["features"]
            self.dataset = features_and_dataset["dataset"]
        else:
            if istrain:
                self.examples = self.processcor.get_train_examples(data_dir=None,filename=self.filename)
                self.features, self.dataset = squad_convert_examples_to_features(self.examples,
                                                                                self.tokenizer,
                                                                                max_seq_length = self.max_seq_length,
                                                                                doc_stride = self.doc_stride,
                                                                                max_query_length = self.max_query_length,
                                                                                is_training=True,
                                                                                return_dataset='pt'
                                                                                )
            else:
                self.examples = self.processcor.get_train_examples(data_dir=None,filename=self.filename)
                self.features,self.dataset = squad_encapsulation_features(self.examples,
                                                                                self.tokenizer,
                                                                                max_seq_length=self.max_seq_length,
                                                                                doc_stride=self.doc_stride,
                                                                                max_query_length=self.max_query_length,
                                                                                is_training=True,
                                                                                return_dataset='pt')
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": self.features, "dataset": self.dataset, "examples": self.examples}, cached_features_file)