import json
import os, re
import torch
from transformers import cached_path
from collections import defaultdict
from itertools import chain
from torch.utils.data import TensorDataset
import random

special_tokens = {'persona1_token':'<persona1>', 'persona2_token':'<persona2>', 'speaker1_token':'<speaker1>', 'speaker2_token':'<speaker2>'}

def add_special_tokens_(model, tokenizer, special_tokens):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # e.g., special_tokens = {'subj_token': '<subj>'}
    # orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    tokenizer.__dict__.update(special_tokens)
    if num_added_tokens > 0:
        print(num_added_tokens, 'tokens are added!')
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return tokenizer, model




def get_dataset(args, tokenizer, train_dataset_path, valid_dataset_path):

    train_dataset_cache = train_dataset_path[:-3] + 'tar.gz_' + type(tokenizer).__name__
    valid_dataset_cache = valid_dataset_path[:-3] + 'tar.gz_' + type(tokenizer).__name__
    if os.path.isfile(train_dataset_cache) and os.path.isfile(valid_dataset_cache):
        train_dataset = torch.load(train_dataset_cache)
        valid_dataset = torch.load(valid_dataset_cache)

    else:
        train_file = cached_path(train_dataset_path)
        valid_file = cached_path(valid_dataset_path)
        file_dict = {"train": train_file, "valid": valid_file}
        pattern = "^[0-9]+"
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

        for name, file in file_dict.items():
            all_data_enc = []
            with open(file, "r", encoding="utf-8") as f:
                dataset = f.readlines()
                data_list = []
                dialogue_list = None
                for line in dataset:
                    if line.startswith('1 '):
                        if dialogue_list is not None:
                            data_list.append(dialogue_list)
                        dialogue_list = []
                        dialogue_list.append(line)
                    else:
                        dialogue_list.append(line)
                for idx, dialogue in enumerate(data_list):
                    dialogue_dict = dict()
                    dial_idx = idx
                    history_list = []
                    utterance_list = []
                    your_persona = []
                    partner_persona = []

                    for line in dialogue:
                        if "your persona:" in line:
                            _, persona = line.split("persona: ")
                            persona = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
                            your_persona.append(persona)

                        elif "partner's persona" in line:
                            #print('partner' , line)
                            _, persona = line.split("persona: ")
                            persona = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
                            partner_persona.append(persona)

                        else:
                            turn_dict = dict()
                            partner_utt, your_utt, _, candidates = line.split('\t')
                            partner_utt = re.sub(pattern, '', partner_utt)
                            partner_utt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(partner_utt.strip()))
                            your_utt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(your_utt.strip()))
                            candidate_list = candidates.split('|')
                            candidate_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(candidate.strip())) for candidate in candidate_list]
                            current_turn = [partner_utt, your_utt]

                            turn_dict['dialogue'] = history_list + current_turn
                            turn_dict['candidate_list'] = candidate_list


                            history_list = history_list + current_turn
                            utterance_list.append(turn_dict)

                    # print('your persona: ', your_persona)
                    # print('partner persona: ', partner_persona)
                    # print('utterance_list: ', utterance_list)
                    dialogue_dict['dial_idx'] = dial_idx
                    dialogue_dict['your_persona'] = your_persona
                    dialogue_dict['partner_persona'] = partner_persona
                    dialogue_dict['utterance'] = utterance_list
                    all_data_enc.append(dialogue_dict)

            print("Tokenize and encode the dataset")

            datasets[name] = all_data_enc


            if name == 'train':
                torch.save(datasets[name], train_dataset_cache)
            else:
                torch.save(datasets[name], valid_dataset_cache)

        train_dataset = datasets['train']
        valid_dataset = datasets['valid']

    if args.few_shot_setting == True:
        if args.few_shot_data == "a":
            random.seed(64)
        elif args.few_shot_data == "b":
            random.seed(4)
        elif args.few_shot_data == "c":
            random.seed(128)
        else:
            raise NotImplementedError
        train_dataset = random.sample(train_dataset, args.few_shot_num)
    return train_dataset, valid_dataset


def get_test_dataset(tokenizer, test_dataset_path):

    test_dataset_cache = test_dataset_path[:-3] + 'tar.gz_' + type(tokenizer).__name__
    if os.path.isfile(test_dataset_cache):
        test_dataset = torch.load(test_dataset_cache)
    else:
        test_file = cached_path(test_dataset_path)
        file_dict = {"test": test_file}
        pattern = "^[0-9]+"
        datasets = {"test": defaultdict(list)}

        for name, file in file_dict.items():
            all_data_enc = []
            with open(file, "r", encoding="utf-8") as f:
                dataset = f.readlines()
                data_list = []
                dialogue_list = None
                for line in dataset:
                    if line.startswith('1 '):
                        if dialogue_list is not None:
                            data_list.append(dialogue_list)
                        dialogue_list = []
                        dialogue_list.append(line)
                    else:
                        dialogue_list.append(line)
                for idx, dialogue in enumerate(data_list):
                    dialogue_dict = dict()
                    dial_idx = idx
                    history_list = []
                    utterance_list = []
                    your_persona = []
                    partner_persona = []

                    for line in dialogue:
                        if "your persona:" in line:
                            _, persona = line.split("persona: ")
                            persona = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
                            your_persona.append(persona)

                        elif "partner's persona" in line:
                            #print('partner' , line)
                            _, persona = line.split("persona: ")
                            persona = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona.strip()))
                            partner_persona.append(persona)

                        else:
                            turn_dict = dict()
                            partner_utt, your_utt, _, candidates = line.split('\t')
                            partner_utt = re.sub(pattern, '', partner_utt)
                            partner_utt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(partner_utt.strip()))
                            your_utt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(your_utt.strip()))
                            candidate_list = candidates.split('|')
                            candidate_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(candidate.strip())) for candidate in candidate_list]
                            current_turn = [partner_utt, your_utt]

                            turn_dict['dialogue'] = history_list + current_turn
                            turn_dict['candidate_list'] = candidate_list


                            history_list = history_list + current_turn
                            utterance_list.append(turn_dict)

                    dialogue_dict['dial_idx'] = dial_idx
                    dialogue_dict['your_persona'] = your_persona
                    dialogue_dict['partner_persona'] = partner_persona
                    dialogue_dict['utterance'] = utterance_list
                    all_data_enc.append(dialogue_dict)

            print("Tokenize and encode the dataset")

            datasets['test'] = all_data_enc

            torch.save(datasets['test'], test_dataset_cache)

        test_dataset = datasets['test']

    return test_dataset


def get_data_loaders(args, tokenizer):
    train_dataset, valid_dataset = get_dataset(args, tokenizer, args.train_dataset_path, args.valid_dataset_path)
    data_dict = {"train": train_dataset, "valid": valid_dataset}
    data = {"train": defaultdict(list), "valid": defaultdict(list)}
    template = tuple([int(item) for item in args.template.split(',')])

    for name, data_value in data_dict.items():
        if args.test_mode == True:
            data_value = data_value[:5]

        for dialog in data_value:
            dial_idx = dialog['dial_idx']
            your_persona = dialog['your_persona']
            partner_persona = dialog['partner_persona']
            utterance = dialog['utterance']
            for utt_item in utterance:
                current_utt = utt_item['dialogue']
                candidate = utt_item['candidate_list']
                instance = build_input_bart(args, tokenizer, dial_idx, your_persona, partner_persona, current_utt, candidate, args.ptuning, args.manual_tuning, template)
                for input_name, input_array in instance.items():
                    data[name][input_name].append(input_array)

    tensor_data = {"train": [], "valid": []}
    for dataset_name, dataset in data.items():
        padded_dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name, input_array in padded_dataset.items():
            tensor = torch.tensor(dataset[input_name], device=args.device)
            print(input_name, tensor.size())
            tensor_data[dataset_name].append(tensor)
    train_dataset, valid_dataset = TensorDataset(*tensor_data["train"]), TensorDataset(*tensor_data["valid"])

    return train_dataset, valid_dataset

def get_testdata_loaders(args, tokenizer):
    test_dataset = get_test_dataset(tokenizer, args.test_dataset_path)
    data_dict = {"test": test_dataset}
    data = {"test": defaultdict(list)}
    template = tuple([int(item) for item in args.template.split(',')])

    for name, data_value in data_dict.items():
        for dialog in data_value:
            dial_idx = dialog['dial_idx']
            your_persona = dialog['your_persona']
            partner_persona = dialog['partner_persona']
            utterance = dialog['utterance']
            for utt_item in utterance:
                current_utt = utt_item['dialogue']
                candidate = utt_item['candidate_list']
                instance = build_input_bart(args, tokenizer, dial_idx, your_persona, partner_persona, current_utt, candidate, args.ptuning, args.manual_tuning, template)
                for input_name, input_array in instance.items():
                    data[name][input_name].append(input_array)

    tensor_data = {"test": []}
    for dataset_name, dataset in data.items():
        padded_dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)

        for input_name, input_array in padded_dataset.items():
            tensor = torch.tensor(dataset[input_name], device=args.device)
            tensor_data[dataset_name].append(tensor)
    test_dataset = TensorDataset(*tensor_data["test"])

    return test_dataset

def build_input_bart(args, tokenizer, dial_idx, your_persona, partner_persona, current_utt, candidate, ptuning=False, manual_tuning=False, template=(3,3,3)):
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    persona1, persona2, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(special_tokens.values()) #persona1: partner, persona2: you, speaker1: partner, speaker2: you


    if len(your_persona) == 0:
        your_persona = None
    else:
        your_persona = list(chain(*[[persona2]] + your_persona))
    if len(partner_persona) == 0:
        partner_persona = None
    else:
        partner_persona = list(chain(*[[persona1]] + partner_persona))

    history, reply = current_utt[:-1], current_utt[-1]
    history = [[persona1 if i % 2 == 0 else persona2] + s for i, s in enumerate(history)]
    reply = [persona2] + reply

    if ptuning == False:
        if manual_tuning == False:
            input_ids = [[bos]] + [partner_persona if partner_persona != None else []] + [your_persona if your_persona != None else []] + history + [[eos]]
        else:
            manual_template = ["Partner's persona sentences are: ", "Your persona sentences are: ", "Persona-based dialogue: "]
            manual_template = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(m_template)) for m_template in manual_template]
            input_ids = [[bos]] + [manual_template[0]] + [partner_persona if partner_persona != None else []] + [manual_template[1]] + [your_persona if your_persona != None else []] + [manual_template[2]] + history + [[eos]]
    else:
        pseudo_token_id = tokenizer.get_vocab()[args.pseudo_token]
        input_ids = [[bos]] + [[pseudo_token_id] * template[0]] + [partner_persona if partner_persona != None else []] + [[pseudo_token_id] * template[1]] + [your_persona if your_persona != None else []] + [[pseudo_token_id] * template[2]] + history + [[eos]]

    input_ids = list(chain(*input_ids))
    decoder_input_ids = [bos] + reply
    labels = decoder_input_ids[1:] + [eos]

    instance = dict()
    instance['input_ids'] = input_ids
    instance['decoder_input_ids'] = decoder_input_ids
    instance['lm_labels'] = labels
    return instance

def pad_dataset(dataset, padding=1):
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
    dataset["input_ids"] = [x + [padding] * (max_enc_l - len(x)) for x in dataset["input_ids"]]
    dataset["decoder_input_ids"] = [x + [padding] * (max_dec_l - len(x)) for x in dataset["decoder_input_ids"]]
    dataset["lm_labels"] = [x + [-100] * (max_dec_l - len(x)) for x in dataset["lm_labels"]]
    return dataset

