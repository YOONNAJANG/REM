import json
import logging
import os
import tarfile
import tempfile
import torch
from transformers import cached_path
from collections import defaultdict
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
logger = logging.getLogger(__file__)

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

GPT2_SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<keyword>", "<pad>"]
GPT2_ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>', "<keyword>"]}
GPT2_MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
GPT2_PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
T5_SPECIAL_TOKENS = ["<speaker1>", "<speaker2>", "<keyword>"]
T5_ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<speaker1>', '<speaker2>', "<keyword>"]}
T5_MODEL_INPUTS = ["input_ids", "decoder_ids", "labels"]
T5_PADDED_INPUTS = ["input_ids", "decoder_ids", "labels"]

def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, data_type):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    if data_type == 'original':
        dataset_path = "/home/yoonna/persona_chat/data/persona-chat/persona_chat_original.json"
        dataset_cache = "cache/cache.tar.gz_original"
    elif data_type == 'revised':
        dataset_path = "/home/yoonna/persona_chat/data/persona-chat/persona_chat_revised.json"
        dataset_cache = "cache/cache.tar.gz_revised"
    elif data_type == 'convai':
        dataset_path = "/home/yoonna/persona_chat/data/convai2/personachat.json"
        dataset_cache = "cache/cache.tar.gz_original"
    elif data_type == 'empchat':
        raise NotImplementedError
    else:
        raise NotImplementedError

    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
        #logger.info("dataset: ", dataset.keys(), len(dataset['test']))
    else:
        logger.info("Download dataset from %s", dataset_path)
        #personachat_file = cached_path(dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
            print(dataset.keys(), len(dataset['test']))

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(dir_name: str):
    logdir = os.path.join(
        '/home/mnt/yoonna/persona_chat', dir_name
    )
    os.mkdir(logdir)
    return logdir

def gpt2_pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in GPT2_PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def t5_pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_dec_l = max(len(x) for x in dataset["decoder_ids"])
    #for name in T5_PADDED_INPUTS:
    dataset["input_ids"] = [x + [padding] * (max_enc_l - len(x)) for x in dataset["input_ids"]]
    dataset["decoder_ids"] = [x + [padding] * (max_dec_l - len(x)) for x in dataset["decoder_ids"]]
    dataset["labels"] = [x + [padding] * (max_dec_l - len(x)) for x in dataset["labels"]]
    return dataset

def add_special_tokens_gpt2(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(GPT2_ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def add_special_tokens(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.get_vocab())
    num_added_tokens = tokenizer.add_special_tokens(T5_ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    print("orig num", orig_num_tokens, "num_added", num_added_tokens) #50265, 4
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
        print("vocab including s.t: ", len(tokenizer.get_vocab()))



def build_input_gpt2(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2, keyword = tokenizer.convert_tokens_to_ids(GPT2_SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def build_input_t5(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    speaker1, speaker2, keyword = tokenizer.convert_tokens_to_ids(T5_SPECIAL_TOKENS)
    #prefix = tokenizer.encode("generate persona dialog:")
    prefix = [3806, 568, 9, 13463, 10]
    sequence = [prefix + list(chain(*persona))] + history
    dec_seq = reply + [1]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])] + [[1]]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["decoder_ids"] = dec_seq
    instance["labels"] = dec_seq
    return instance

def build_input_bart(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    speaker1, speaker2, keyword = tokenizer.convert_tokens_to_ids(T5_SPECIAL_TOKENS)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    #prefix = tokenizer.encode("generate persona dialog:")
    sequence = [[bos] + list(chain(*persona))] + history
    dec_seq = reply + [eos]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["decoder_ids"] = dec_seq
    instance["labels"] = dec_seq
    return instance


def get_gpt2_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.data_type)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        #if args.num_candidates > 0 and dataset_name == 'train':
        if args.num_candidates > 0:
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_gpt2(persona, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = gpt2_pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(GPT2_SPECIAL_TOKENS[-1]))
        for input_name in GPT2_MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler

def get_t5_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.data_type)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        if dataset_name == 'test':
            continue
        #logger.info("data: ", dataset_name, len(datasets[dataset_name]), datasets[dataset_name][0])
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        #if args.num_candidates > 0 and dataset_name == 'train':
        if args.num_candidates > 0:
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_t5(persona, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                #datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                #datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = t5_pad_dataset(dataset, padding=0)
        for input_name in T5_MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            #if input_name != "mc_labels":
            #    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler

def get_bart_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.data_type)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        if dataset_name == 'test':
            continue
        #logger.info("data: ", dataset_name, len(datasets[dataset_name]), datasets[dataset_name][0])
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        #if args.num_candidates > 0 and dataset_name == 'train':
        if args.num_candidates > 0:
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_bart(persona, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                #datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                #datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = t5_pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in T5_MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            #if input_name != "mc_labels":
            #    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler