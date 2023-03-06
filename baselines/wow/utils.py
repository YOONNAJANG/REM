import copy
import json
import logging
import os
import torch
from transformers import cached_path
from itertools import chain
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__file__)

special_tokens = {'wizard_token':'<wizard>', 'apprentice_token':'<apprentice>', 'persona_token':'<persona>', 'knowledge_token':'<knowledge>'}


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

def pad_dataset(dataset, padding=1):
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
    # max_per_l = max(len(x) for x in dataset["persona"])
    new_dataset = dict()
    new_dataset['input_ids'] = [x + [padding] * (max_enc_l - len(x)) for x in dataset['input_ids']]
    new_dataset['decoder_input_ids'] = [x + [padding] * (max_dec_l - len(x)) for x in dataset['decoder_input_ids']]
    new_dataset['lm_labels'] = [x + [-100] * (max_dec_l - len(x)) for x in dataset['lm_labels']]
    # new_dataset['persona'] = [x + [padding] * (max_per_l - len(x)) for x in dataset['persona']]
    return new_dataset



def build_input_for_bart(args, history, checked_sentences, persona, tokenizer):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    wizard_st = tokenizer.convert_tokens_to_ids(tokenizer.wizard_token)
    apprentice_st = tokenizer.convert_tokens_to_ids(tokenizer.apprentice_token)
    persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)

    # print(f"history[0][0]: ({tokenizer.decode(history[0][0])})")
    if tokenizer.decode(history[0][0]) == "0_Wizard":   #wizard starts
        history = history[1:]
        checked_sentences = checked_sentences[1:]
    
    history_list = []
    history_data = []
    for i, each_utterance in enumerate(history):
        history_list.append(each_utterance)
        if i % 2 == 1:
            history_now = copy.deepcopy(history_list)
            history_data.append(history_now)

    input_list = list()
    for history, knowledge in zip(history_data, checked_sentences):
        dial_dict = {}
        reply = history[-1][-1]
        dial_hist = history[-(2*args.max_history+2):-1]
        
        # print(f"dial_hist[0][0]: ({tokenizer.decode(dial_hist[0][0])})")

        dialogue_history = [[apprentice_st]+utt[-1] if i%2==0 else [wizard_st]+utt[-1] for i, utt in enumerate(dial_hist)]
        # input_ids = [[bos] + [knowledge_st]] + [knowledge] + [[persona_st]] + [persona] + dialogue_history + [[eos]]
        input_ids = [[bos] + [knowledge_st]] + [knowledge] + dialogue_history + [[eos]]
        dial_dict['input_ids'] = list(chain(*input_ids))
        dial_dict['decoder_input_ids'] = [bos] + reply
        dial_dict["lm_labels"] = reply + [eos]
        # dial_dict["persona"] = persona

        input_list.append(dial_dict)

    return input_list


def get_dataset_only_train_dev(tokenizer, train_dataset_path, dev_dataset_path):

    no_knowledge_count = 0
    knowledge_count = 0
    app_knowledge_count = 0

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    train_dataset_cache = train_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
    # train_dataset_cache = train_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
    # dev_dataset_cache = dev_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
    #

    print(train_dataset_cache)
    if train_dataset_cache and os.path.isfile(train_dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        logger.info("Process dataset from %s", train_dataset_path)
        wow_file_train = cached_path(train_dataset_path)
        wow_file_dev = cached_path(dev_dataset_path)
        # file_dict = {"train": wow_file_train, "valid": wow_file_dev}
        file_dict = {"valid": wow_file_dev, "train": wow_file_train}
        all_dataset = dict()
        for name, file in file_dict.items():
            print(name, file)
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()

                for data in dataset:
                    persona = data["persona"] # sentence
                    new_dialogue = {"dialog": []}
                    for each_dialog in data["dialog"]:
                        utt_enc = dict()
                        speaker = each_dialog["speaker"]    # 0_Wizard, 1_Wizard, 0_Apprentice, 1_Apprentice
                        text = each_dialog["text"] # utterance

                        if "checked_sentence" in each_dialog.keys():
                            if each_dialog["speaker"].split("_")[1] != "Wizard":
                                print('each_dialog["speaker"].split("_")[1] == "Wizard"')
                                exit()

                            if len(each_dialog["checked_sentence"]) > 1:
                                print(len(each_dialog["checked_sentence"]))
                                exit()

                            if ('no_passages_used' in each_dialog["checked_sentence"].keys()) or (len(each_dialog["checked_sentence"]) == 0):
                                checked_sentence = 'no_passages_used'
                                no_knowledge_count += 1
                            elif ('no_passages_used' not in each_dialog["checked_sentence"].keys()) and (len(each_dialog["checked_sentence"]) == 1):
                                checked_sentence = list(each_dialog["checked_sentence"].values())[0]
                                knowledge_count += 1
                            else:
                                print('no_passages_used ?, len(each_dialog["checked_sentence"]) ?')
                                exit()

                        else:
                            if each_dialog["speaker"].split("_")[1] != "Apprentice":
                                print('each_dialog["speaker"].split("_")[1] != "Apprentice"')
                                exit()

                            checked_sentence = 'no_passages_used'
                            app_knowledge_count += 1

                        utt_enc["speaker"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(speaker.strip()))
                        utt_enc["text"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text.strip()))
                        utt_enc["checked_sentence"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(checked_sentence.strip()))

                        new_dialogue["dialog"].append(utt_enc)
                    new_dialogue["persona"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            #
            # breakpoint()

            print("no_knowledge_count:", no_knowledge_count)
            print("knowledge_count:", knowledge_count)
            print("app_knowledge_count:", app_knowledge_count)

            if name == 'train':
                print('saving train')
                torch.save(dataset, train_dataset_cache)
            else:
                print('saving valid')
                torch.save(dataset, dev_dataset_cache)
    return all_dataset


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """

    focus = get_dataset_only_train_dev(tokenizer, args.train_dataset_path, args.dev_dataset_path)

    model_name = args.model_name

    logger.info("Build inputs and labels")
    # datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    datasets = {"valid": defaultdict(list), "train": defaultdict(list)}

    for dataset_name, dataset in focus.items():
        print(dataset_name, len(dataset))
        for data in dataset:

            persona = data['persona']
            checked_sentences = []
            history_sent = []
            history = []

            for each_dialog in data['dialog']:
                history_sent.append(tokenizer.decode(each_dialog['text']))
                history.append([each_dialog['speaker'], each_dialog['text']])
                checked_sentences.append(each_dialog['checked_sentence'])

            if model_name == 'BART' or model_name == 'transformer-encdec':
                    instance_list = build_input_for_bart(args, history, checked_sentences, persona, tokenizer)

            elif model_name == 'T5':
                    instance_list = build_input_for_t5(args, history, checked_sentences, persona, tokenizer)

            for instance in instance_list:
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in ['input_ids', 'decoder_input_ids', 'lm_labels']:  #, 'persona']:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)


    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])

    return train_dataset, valid_dataset


# def get_dataset_only_train_dev(tokenizer, train_dataset_path, dev_dataset_path):
#
#     def tokenize(obj):
#         if isinstance(obj, str):
#             return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
#         if isinstance(obj, dict):
#             return dict((n, tokenize(o)) for n, o in obj.items())
#         return list(tokenize(o) for o in obj)
#
#     train_dataset_cache = train_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
#     dev_dataset_cache = dev_dataset_path[:-5] + '_wow_' + type(tokenizer).__name__
#     tmp_count = 0
#     if dev_dataset_cache and os.path.isfile(dev_dataset_cache):
#         logger.info("Load tokenized dataset from cache at %s", train_dataset_cache)
#         train_dataset = torch.load(train_dataset_cache)
#         dev_dataset = torch.load(dev_dataset_cache)
#         all_dataset = dict()
#         all_dataset["train"] = train_dataset["train"]
#         all_dataset["valid"] = dev_dataset["valid"]
#     else:
#         logger.info("Process dataset from %s", train_dataset_path)
#         wow_file_train = cached_path(train_dataset_path)
#         wow_file_dev = cached_path(dev_dataset_path)
#         # file_dict = {"train": wow_file_train, "valid": wow_file_dev}
#         file_dict = {"valid": wow_file_dev, "train": wow_file_train}
#         all_dataset = dict()
#         for name, file in file_dict.items():
#             print(name, file)
#             with open(file, "r", encoding="utf-8") as f:
#                 dataset = json.loads(f.read())
#                 dataset_enc = dict()
#                 dataset_enc[name] = list()
#                 for dialogue in dataset["data"]:
#                     ID = dialogue["dialogID"]
#                     persona = dialogue["persona"]
#                     # knowledge = dialogue["knowledge"]
#                     utterance = dialogue["utterance"]
#
#                     new_dialogue = dict()
#                     new_dialogue["dialog"] = list()
#                     for i, utt in enumerate(utterance):
#                         key = "dialogue" + str(i+1)
#                         dial = utt[key]
#                         dial_new = dict()
#                         dial_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in dial]
#                         # persona_ground_enc = [1 if item==True else 0 for item in persona_ground]
#                         # knowledge_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge_can]
#                         # persona_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona_can]
#                         dial_new["text"] = dial_enc
#                         # dial_new["dialog"] = dial_enc[-1]
#                         # dial_new["persona_candidates"] = persona_can_enc
#                         # dial_new["persona_grounding"] = persona_ground_enc
#                         # dial_new["knowledge_candidates"] = knowledge_can_enc
#                         # dial_new["knowledge_answer_index"] = knowledge_answer
#
#                         new_dialogue["dialog"].append(dial_new)
#                         tmp_count += len(dial_new["text"])
#
#                     new_dialogue["persona"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
#                     dataset_enc[name].append(new_dialogue)
#
#         logger.info("Tokenize and encode the dataset")
#
#         dataset = dataset_enc
#         all_dataset[name] = dataset_enc[name]
#         # if name == 'train':
#         #     print('saving train')
#         #     torch.save(dataset, train_dataset_cache)
#         # else:
#         #     print('saving valid')
#         #     torch.save(dataset, dev_dataset_cache)
#     return dataset



#
#
# def get_data_loaders(args, tokenizer):
#     """ Prepare the dataset for training and evaluation """
#
#     focus = get_dataset_only_train_dev(tokenizer, args.train_dataset_path, args.dev_dataset_path)
#
#     model_name = args.model_name
#
#     logger.info("Build inputs and labels")
#     datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
#
#     for dataset_name, dataset in focus.items():
#         print(dataset_name, len(dataset))
#         for data in dataset:
#
#             utterance = data['dialog']
#             persona = data['persona']
#             checked_sentences = []
#             print(utterance)
#             breakpoint()
#             for i, utt in enumerate(utterance):
#                 history = utt["text"][-2:]
#
#                 if model_name == 'BART' or model_name == 'transformer-encdec':
#                         instance = build_input_for_bart(args, history, checked_sentences, persona, tokenizer)
#
#                 elif model_name == 'T5':
#                         instance = build_input_for_t5(args, history, checked_sentences, persona, tokenizer)
#
#
#                 for input_name, input_array in instance.items():
#                     datasets[dataset_name][input_name].append(input_array)
#
#     logger.info("Pad inputs and convert to Tensor")
#     tensor_datasets = {"train": [], "valid": []}
#
#     for dataset_name, dataset in datasets.items():
#         dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
#         for input_name in ['input_ids', 'decoder_input_ids', 'lm_labels', 'persona']:
#             tensor = torch.tensor(dataset[input_name], device=args.device)
#             print(input_name, tensor.size())
#             tensor_datasets[dataset_name].append(tensor)
#
#
#     logger.info("Build train and validation dataloaders")
#     train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
#
#     return train_dataset, valid_dataset