import copy
import json
import logging
import os
import torch
from transformers import cached_path
from itertools import chain
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__file__)

special_tokens = {'query_token':'<query>', 'answer_token':'<answer>', 'knowledge_token':'<knowledge>'}

reply_max_len = -1
reply_max_tokens = None
reply_over_flag = False

def add_special_tokens_(model, tokenizer, special_tokens):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # e.g., special_tokens = {'subj_token': '<subj>'}
    # orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    tokenizer.__dict__.update(special_tokens)
    if num_added_tokens > 0:
        print(num_added_tokens, 'tokens are added!')
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return model, tokenizer


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """

    focus = get_dataset_only_train_dev(tokenizer, args.train_dataset_path, args.dev_dataset_path)

    model_name = args.model_name

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in focus.items():
        print(dataset_name, len(dataset))
        for data in tqdm(dataset):
            if model_name == 'BART' or model_name == 'transformer-encdec':
                    instance_list = build_input_for_bart(args, data['dialog'], tokenizer)

            elif model_name == 'T5':
                    instance_list = build_input_for_t5(args, data['dialog'], tokenizer)

            for instance in instance_list:
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in ['input_ids', 'decoder_input_ids', 'lm_labels', 'knowledge']:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)


    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])

    global reply_max_len
    global reply_max_tokens
    print("\nreply_max_len:", reply_max_len)
    print("\nreply_max_tokens:", reply_max_tokens)

    return train_dataset, valid_dataset


def get_dataset_only_train_dev(tokenizer, train_dataset_path, dev_dataset_path):

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    train_dataset_cache = train_dataset_path[:-5] + '_cmudog_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_path[:-5] + '_cmudog_' + type(tokenizer).__name__

    if (train_dataset_cache and os.path.isfile(train_dataset_cache)) and (dev_dataset_cache and os.path.isfile(dev_dataset_cache)):
        logger.info("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        logger.info("Process dataset from %s", train_dataset_path)
        cmudog_file_train = cached_path(train_dataset_path)
        cmudog_file_dev = cached_path(dev_dataset_path)
        # file_dict = {"train": cmudog_file_train, "valid": cmudog_file_dev}
        file_dict = {"valid": cmudog_file_dev, "train": cmudog_file_train}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for data in tqdm(dataset["data"]):
                    new_dialogue = dict()
                    new_dialogue["dialog"] = list()
                    for i, utt in enumerate(data["utterance"]):
                        utt_enc = dict()
                        for utt_k, utt_v in utt.items():
                            if type(utt_v) is list:
                                utt_enc[utt_k] = []
                                for each_history_sent in utt_v:
                                    utt_enc[utt_k].append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(each_history_sent.strip())))
                            elif type(utt_v) is str:
                                utt_enc[utt_k] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt_v.strip()))
                            else:
                                print(f"type(utt_v) is {type(utt_v)}")
                                exit()
                        new_dialogue["dialog"].append(utt_enc)
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                print('saving train')
                torch.save(dataset, train_dataset_cache)
            else:
                print('saving valid')
                torch.save(dataset, dev_dataset_cache)
    return all_dataset




def build_input_for_bart(args, history_knowledge_data, tokenizer):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    query_st = tokenizer.convert_tokens_to_ids(tokenizer.query_token)
    answer_st = tokenizer.convert_tokens_to_ids(tokenizer.answer_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)

    # history_data = []
    # history_list = []
    # if history[0][0] == [288, 1215, 771, 44417]: #wizard starts
    #     history = history[:-1]
    # for i, utt in enumerate(history):
    #     history_list.append(utt)
    #     tokenizer = tokenizer
    #     if utt[0] == [288, 1215, 771, 44417]:
    #         history_now = copy.deepcopy(history_list)
    #         history_data.append(history_now)

    global reply_max_len
    global reply_max_tokens
    global reply_over_flag

    input_list = list()
    for history_knowledge in history_knowledge_data:
        dial_dict = {}
        tokenizer = tokenizer
        for utt_k, utt_v in history_knowledge.items():
            if utt_k != "selected_knowledge":
                reply = history_knowledge[utt_k][-1]

                # if len(reply) > 112:
                #     reply_over_flag = True
                # else:
                #     reply_over_flag = False

                if len(reply) > reply_max_len:
                    reply_max_len = len(reply)
                    reply_max_tokens = reply
                dial_hist = history_knowledge[utt_k][-(2*args.max_history+2):-1]
                break
        
        if reply_over_flag is True:
            continue
        
        if len(dial_hist) == 0:
            # dialogue_history = [[wizard_st] + utt[-1] if i % 2 == 0 else [apprentice_st] + utt[-1] for i, utt in enumerate(dial_hist)]
            input_ids = [[bos] + [knowledge_st]] + [history_knowledge["selected_knowledge"]] + [[eos]]
            dial_dict['input_ids'] = list(chain(*input_ids))
            dial_dict['decoder_input_ids'] = [bos] + reply
            dial_dict["lm_labels"] = reply + [eos]
            dial_dict["knowledge"] = history_knowledge["selected_knowledge"]
            print("len(dial_hist) == 0")
            exit()

        else:
            dialogue_history = [[query_st]+utt if i%2==0 else [answer_st]+utt for i, utt in enumerate(dial_hist)]
            input_ids = [[bos] + [knowledge_st]] + [history_knowledge["selected_knowledge"]] + dialogue_history + [[eos]]

            if len(list(chain(*input_ids))) > 1024:
                continue

            dial_dict['input_ids'] = list(chain(*input_ids))
            dial_dict['decoder_input_ids'] = [bos] + reply
            dial_dict["lm_labels"] = reply + [eos]
            dial_dict["knowledge"] = history_knowledge["selected_knowledge"]

        input_list.append(dial_dict)
    return input_list


def pad_dataset(dataset, padding=1):
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
    max_kwg_l = max(len(x) for x in dataset["knowledge"])
    new_dataset = dict()
    new_dataset['input_ids'] = [x + [padding] * (max_enc_l - len(x)) for x in dataset['input_ids']]
    new_dataset['decoder_input_ids'] = [x + [padding] * (max_dec_l - len(x)) for x in dataset['decoder_input_ids']]
    new_dataset['lm_labels'] = [x + [-100] * (max_dec_l - len(x)) for x in dataset['lm_labels']]
    new_dataset['knowledge'] = [x + [padding] * (max_kwg_l - len(x)) for x in dataset['knowledge']]
    return new_dataset