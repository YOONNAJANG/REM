import logging
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import copy
from collections import Counter
from tqdm import tqdm
from transformers import cached_path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from baselines.FoCus.utils_focus import get_dataset_refine

import json
import random
from random import randrange


special_tokens_focus = {'machine_token':'<machine>', 'human_token':'<human>', 'persona_token':'<persona>', 'knowledge_token':'<knowledge>'}
# special_tokens_focus = {'machine_token':50265, 'human_token':50266, 'persona_token':50267, 'knowledge_token':50268}

MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "ner_labels"]
logger = logging.getLogger(__file__)

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

def add_special_tokens_test(model, congen_model, tokenizer, special_tokens):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # e.g., special_tokens = {'subj_token': '<subj>'}
    # orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    tokenizer.__dict__.update(special_tokens)
    if num_added_tokens > 0:
        print(num_added_tokens, 'tokens are added!')
        print(len(tokenizer))
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        congen_model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return tokenizer, model, congen_model



def build_input_focus(args, tokenizer, history, persona_cans, persona_ner_label, golden_knowledge, knowledge_ner_label, template=(3,3,3)): #gk|p|history|u' -> u
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    dec_bos = 2  # tokenizer.decoder_start_token_id
    # input_ids = [bos] + gt_knowledge + [eos] + corrputed[0] + [eos]
    # machine_st = tokenizer.convert_tokens_to_ids(tokenizer.machine_token)
    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
    # persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    # knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)

    history_, reply = history[:-1], history[-1]

    gold_knowledge = golden_knowledge  # bos 포함
    knowledge_label = []
    knowledge_label.extend(knowledge_ner_label)
    history_ = [human_st] + history_[0]
    assert len(persona_ner_label) == len(persona_cans)


    if args.ptuning == False:
        enc_sequence = gold_knowledge + persona_cans
        enc_sequence.extend(history_)
        ner_label = knowledge_label + persona_ner_label

    else:
        pseudo_token_id = tokenizer.get_vocab()[args.pseudo_token]
        enc_sequence = [bos] + [pseudo_token_id] * template[0] + gold_knowledge[1:] + [pseudo_token_id] * template[
            1] + persona_cans + [pseudo_token_id] * template[2]
        enc_sequence.extend(history_)
        ner_label = [-1] + [-1] * template[0] + knowledge_label[1:] + [-1] * template[1] + persona_ner_label + [-1] * \
                    template[2]

    dec_sequence = [dec_bos] + reply + [eos]

    instance = dict()
    instance['input_ids'] = enc_sequence  # [bos] [knoweldge token] gk [persona token] ps [human token] history(last)
    instance['decoder_input_ids'] = dec_sequence[:-1]
    instance['lm_labels'] = dec_sequence[1:]
    instance['ner_labels'] = ner_label
    return instance


def build_input_wow(tokenizer, history, persona_cans, golden_knowledge, knowledge_ner_label): #gk|p|history|u' -> u
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    dec_bos = 2  # tokenizer.decoder_start_token_id
    # input_ids = [bos] + gt_knowledge + [eos] + corrputed[0] + [eos]
    machine_st = tokenizer.convert_tokens_to_ids(tokenizer.machine_token)
    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
    persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)

    history_, reply = history[:-1], history[-1]

    # history_list = history
    # history_context, history_question = history_list[:-1], history_list[-1]
    # if len(history_context) == 0:
    #     history_context = [1]
    # else:
    #     history_context = list(chain(*history_context))
    #
    # history_list_new = []
    # history_list_new.append(history_context)
    # history_list_new.append(history_question)
    #
    # history = [human_st if i % 2 == 0 else machine_st + s for i, s in enumerate(history)]

    gold_knowledge =golden_knowledge
    knowledge_label = []
    knowledge_label.extend(knowledge_ner_label)
    # print(len(gold_knowledge), gold_knowledge)
    # print(len(knowledge_label),knowledge_label)
    history_ = [human_st] + history_[0]

    # if len(history) == 1:
        # enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history
    enc_sequence = gold_knowledge + persona_cans
    enc_sequence.extend(history_)

    dec_sequence = [dec_bos] + reply + [eos]
    #
    # else:
    #     # enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
    #     enc_sequence = gold_knowledge +persona_cans
    #     enc_sequence.extend(list(chain(*history)))
    #     dec_sequence = [dec_bos] + reply + [eos]
    # # print(enc_sequence)

    # assert len(persona_ner_label) == len(persona_cans)

    ner_label = [bos] + knowledge_label
    # print(dec_sequence)
    # print(ner_label)
    # print()
    ###### # special_tokens_focus = {'machine_token':50265, 'human_token':50266, 'persona_token':50267, 'knowledge_token':50268}
    instance = dict()
    instance['input_ids'] = enc_sequence # [bos] [knoweldge token] gk [persona token] ps [human token] history(last)
    # instance['input_ids'] = list(chain(*enc_sequence))
    instance['decoder_input_ids'] = dec_sequence[:-1]
    instance['lm_labels'] = dec_sequence[1:]
    instance['ner_labels'] = ner_label
    # print(instance)
    return instance

def dataloader_focus(args, tokenizer):

    train_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/train.json"
    train_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/our_train_cache.tar.gz"
    dev_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/dev.json"
    dev_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/our_dev_cache.tar.gz"

    regen_data = get_dataset_refine_focus(tokenizer, train_dataset_path=train_dataset_path,
                                    train_dataset_cache=train_dataset_cache,
                                    dev_dataset_path=dev_dataset_path,
                                    dev_dataset_cache=dev_dataset_cache)

    template = tuple([int(item) for item in args.template.split(',')])

    print("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for (key, value) in regen_data.items():
        if key == 'train' and args.fewshot == True:
            random.shuffle(value)
            value = value[:args.fewnum]
            print(f"Load only {len(value)} data for few-shot experiment.")
        for data in tqdm(value):  # ['dialogID', 'landmark_link', 'replace_history', 'label', 'golden_knowledge', 'human_question', 'machine_ori_answer', 'split_machine_ori_answer', 'split_machine_rep_answer', 'rep_index']
            # dialogID = data['dialogID']
            # persona = data['persona']
            utterance = data['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2 * args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_ner_label = utt['persona_ner_label']
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_focus(args, tokenizer, history, persona_cans, persona_ner_label,
                                             golden_knowledge,
                                             knowledge_ner_label, template)
            for input_name, input_array in instance.items():
                datasets[key][input_name].append(input_array)

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    # print(datasets)
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_focus(dataset, padding=tokenizer.pad_token_id)
        # print(dataset)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    print("Build train and valid dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    return train_dataset, valid_dataset

def dataloader_focus_test(args, tokenizer):

    test_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json"
    test_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/our_test_cache.tar.gz"


    regen_data = get_dataset_refine_focus_test(tokenizer, test_dataset_path=test_dataset_path,
                                    test_dataset_cache=test_dataset_cache)

    template = tuple([int(item) for item in args.template.split(',')])

    print("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    for (key, value) in regen_data.items():
        if key == 'train' and args.fewshot == True:
            random.shuffle(value)
            value = value[:args.fewnum]
        for data in tqdm(value):  # ['dialogID', 'landmark_link', 'replace_history', 'label', 'golden_knowledge', 'human_question', 'machine_ori_answer', 'split_machine_ori_answer', 'split_machine_rep_answer', 'rep_index']
            dialogID = data['dialogID']
            persona = data['persona']
            utterance = data['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2 * args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_ner_label = utt['persona_ner_label']
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_focus(args, tokenizer, history, persona_cans, persona_ner_label,
                                             golden_knowledge,
                                             knowledge_ner_label, template)
            for input_name, input_array in instance.items():
                datasets[key][input_name].append(input_array)
    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}
    # print(datasets)
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_focus(dataset, padding=tokenizer.pad_token_id)
        # print(dataset)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    print("Build test dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])
    return test_dataset

################################################################################################################
################################################################################################################
def dataloader_wow(args, tokenizer):

    train_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_train.json"

    train_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_train_cache.tar.gz"
    dev_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_valid_random_split.json"
    dev_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_dev_cache.tar.gz"

    regen_data = get_dataset_refine_wow(tokenizer, train_dataset_path=train_dataset_path,
                                    train_dataset_cache=train_dataset_cache,
                                    dev_dataset_path=dev_dataset_path,
                                    dev_dataset_cache=dev_dataset_cache)

    print("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for (key, value) in regen_data.items():

        for data in value:  # ['dialogID', 'landmark_link', 'replace_history', 'label', 'golden_knowledge', 'human_question', 'machine_ori_answer', 'split_machine_ori_answer', 'split_machine_rep_answer', 'rep_index']
            dialogID = data['dialogID']
            persona = data['persona']
            utterance = data['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2 * args.max_history):]
                persona_cans = utt['persona_candidates']
                # persona_ner_label = utt['persona_ner_label']
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_wow(tokenizer, history, persona_cans, golden_knowledge,
                                             knowledge_ner_label)
            for input_name, input_array in instance.items():
                datasets[key][input_name].append(input_array)

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    # print(datasets)
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_focus(dataset, padding=tokenizer.pad_token_id)
        # print(dataset)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    print("Build train and valid dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    return train_dataset, valid_dataset




def pad_dataset_focus(dataset, padding):
    max_l = max(len(x) for x in dataset["input_ids"])
    # max_ner_l = max(len(x) for x in dataset["ner_labels"])
    max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
    dataset['input_ids'] = [x + [padding] * (max_l - len(x)) for x in dataset['input_ids']]
    dataset['ner_labels'] = [x + [-1] * (max_l - len(x)) for x in dataset['ner_labels']]
    dataset['decoder_input_ids'] = [x + [padding] * ((max_dec_l) - len(x)) for x in dataset['decoder_input_ids']]
    dataset['lm_labels'] = [x + [-100] * (max_dec_l - len(x)) for x in dataset['lm_labels']]
    return dataset

def get_dataset_refine_focus_test(tokenizer, test_dataset_path, test_dataset_cache):
    ner_label_map = {"B":1, "I":2,"O":0, tokenizer.persona_token:3,tokenizer.knowledge_token:4, tokenizer.bos_token:5} ### knowledge_st, persona_st, bos
    ner_label_count = {0:0, 1:0,2:0}
    token_char = tokenizer.convert_ids_to_tokens(5)[0]
    # print(token_char)
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    test_dataset_cache = test_dataset_cache + '_test_' + type(tokenizer).__name__


    if test_dataset_cache and os.path.isfile(test_dataset_cache):
        print("Load tokenized dataset from cache at %s", test_dataset_cache)
        test_dataset = torch.load(test_dataset_cache)
        all_dataset = dict()
        all_dataset["test"] = test_dataset["test"]

    else:
        print("Process dataset from %s", test_dataset_path)
        file_test = cached_path(test_dataset_path)

        file_dict = {"test": file_test}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    persona = dialogue["persona"]
                    # knowledge = dialogue["knowledge"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        key = "dialogue" + str(i + 1)
                        dial = utt[key]
                        dial_new = dict()
                        persona_can = utt["persona_candidate"]
                        persona_ground = utt["persona_grounding"]
                        knowledge_can = utt["knowledge_candidates"]
                        knowledge_answer = utt["knowledge_answer_index"]
                        knowledge_sent = knowledge_can[knowledge_answer]

                        persona_can_enc = [tokenizer(sentence, add_special_tokens=False) for sentence in persona_can]
                        persona_ground_enc = [1 if item == True else 0 for item in persona_ground]

                        persona_ner_labels = []
                        for i in range(5):
                            persona_ner_labels.append(["O"] * len(persona_can_enc[i]['input_ids']))

                        for ner_label in utt["NER_tagging"].keys():  # LOC, MISC, PER, ORG
                            tmp_persona_index = utt["NER_tagging"][ner_label]["persona_index"]  # p_1부터 p_5까지
                            # print(tmp_persona_index)
                            for i in range(5):
                                if tmp_persona_index[f"p_{i + 1}"] != []:  # persona candidate에 태깅된 entity가 있는 경우
                                    for p in range(len(tmp_persona_index[f"p_{i + 1}"])):
                                        start, end = tmp_persona_index[f"p_{i + 1}"][p]
                                        keyword = persona_can[i][start:end]
                                        start_token_id = persona_can_enc[i].char_to_token(start)
                                        end_token_id = persona_can_enc[i].char_to_token(end - 1)
                                        if start_token_id == None or end_token_id == None:
                                            continue
                                        persona_ner_labels[i][start_token_id] = "B"
                                        persona_ner_labels[i][start_token_id + 1:end_token_id + 1] = ["I"] * (
                                                    end_token_id - start_token_id)

                        ############################# knowledge NER ############################# knowledge NER

                        knowledge_can_enc = tokenizer(knowledge_sent, add_special_tokens=False)
                        knowledge_ner_labels = ["O"] * len(knowledge_can_enc['input_ids'])

                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            # print("NER_LABEL: ", ner_label)
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                keyword = knowledge_sent[start:end]
                                start_token_id = knowledge_can_enc.char_to_token(start)
                                end_token_id = knowledge_can_enc.char_to_token(end - 1)
                                if start_token_id == None or end_token_id == None:
                                    continue
                                knowledge_ner_labels[start_token_id] = "B"
                                knowledge_ner_labels[start_token_id + 1:end_token_id + 1] = ["I"] * (
                                            end_token_id - start_token_id)

                        persona_can_enc_new = []
                        persona_ner_labels_enc = []
                        for (can, ner_label) in zip(persona_can_enc, persona_ner_labels):
                            persona_can_enc_new.append(
                                [tokenizer.convert_tokens_to_ids(tokenizer.persona_token)] + can['input_ids'])
                            persona_ner_labels_enc.append([tokenizer.persona_token] + ner_label)

                        knowledge_can_enc = [tokenizer.bos_token_id,
                                             tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)] + \
                                            knowledge_can_enc['input_ids']
                        knowledge_ner_labels = [tokenizer.bos_token, tokenizer.knowledge_token] + knowledge_ner_labels
                        persona_can_enc = list(chain(*persona_can_enc_new))
                        persona_ner_labels_enc = [[ner_label_map[label] for label in sent] for sent in
                                                  persona_ner_labels_enc]
                        persona_ner_labels_enc = list(chain(*persona_ner_labels_enc))
                        knowledge_ner_labels_enc = [ner_label_map[label] for label in knowledge_ner_labels]

                        for i, l in enumerate(persona_ner_labels_enc):
                            if l in [3, 4, 5]:
                                persona_ner_labels_enc[i] = -1
                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3, 4, 5]:
                                knowledge_ner_labels_enc[i] = -1
                        if type(utt["output"]) == list:
                            pred = utt["output"][0]
                        else:
                            pred = utt["output"]
                        dial[-2] = pred

                        dial_enc = [tokenizer(sentence.strip(), add_special_tokens=False)['input_ids'] for sentence in
                                    dial]

                        assert len(persona_can_enc) == len(persona_ner_labels_enc)
                        assert len(knowledge_can_enc) == len(knowledge_ner_labels_enc)

                        dial_new["dialog"] = dial_enc
                        dial_new["persona_grounding"] = persona_ground_enc
                        dial_new["persona_candidates"] = persona_can_enc
                        dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc

                        new_dialogue["utterance"].append(dial_new)
                    persona_enc = persona_can_enc
                    # knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                    new_dialogue["persona"] = persona_enc
                    # new_dialogue["knowledge"] = knowledge_enc
                    new_dialogue["dialogID"] = ID
                    new_dialogue["landmark_link"] = dialogue["landmark_link"]  ##############################
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            torch.save(dataset, test_dataset_cache)
            # if name == 'train':
            #     torch.save(dataset, train_dataset_cache)
            # else:
            #     torch.save(dataset, dev_dataset_cache)
    return all_dataset


def get_dataset_refine_focus(tokenizer, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):
    ner_label_map = {"B": 1, "I": 2, "O": 0, tokenizer.persona_token: 3, tokenizer.knowledge_token: 4,
                     tokenizer.bos_token: 5}  ### knowledge_st, persona_st, bos

    token_char = tokenizer.convert_ids_to_tokens(5)[0]

    # print(token_char)
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    train_dataset_cache = train_dataset_cache + '_train_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_cache + '_dev_' + type(tokenizer).__name__

    if train_dataset_cache and os.path.isfile(train_dataset_cache):
        print("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        print("Process dataset from %s", train_dataset_path)
        file_train = cached_path(train_dataset_path)
        file_dev = cached_path(dev_dataset_path)
        file_dict = {"train": file_train, "valid": file_dev}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    persona = dialogue["persona"]
                    # knowledge = dialogue["knowledge"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        key = "dialogue" + str(i + 1)
                        dial = utt[key]
                        dial_new = dict()
                        persona_can = utt["persona_candidate"]
                        persona_ground = utt["persona_grounding"]
                        knowledge_can = utt["knowledge_candidates"]
                        knowledge_answer = utt["knowledge_answer_index"]
                        knowledge_sent = knowledge_can[knowledge_answer]

                        persona_can_enc = [tokenizer(sentence, add_special_tokens=False) for sentence in persona_can]
                        persona_ground_enc = [1 if item == True else 0 for item in persona_ground]

                        persona_ner_labels = []
                        for i in range(5):
                            persona_ner_labels.append(["O"] * len(persona_can_enc[i]['input_ids']))

                        for ner_label in utt["NER_tagging"].keys():  # LOC, MISC, PER, ORG
                            tmp_persona_index = utt["NER_tagging"][ner_label]["persona_index"]  # p_1부터 p_5까지
                            # print(tmp_persona_index)
                            for i in range(5):
                                if tmp_persona_index[f"p_{i + 1}"] != []:  # persona candidate에 태깅된 entity가 있는 경우
                                    for p in range(len(tmp_persona_index[f"p_{i + 1}"])):
                                        start, end = tmp_persona_index[f"p_{i + 1}"][p]
                                        keyword = persona_can[i][start:end]
                                        start_token_id = persona_can_enc[i].char_to_token(start)
                                        end_token_id = persona_can_enc[i].char_to_token(end - 1)
                                        if start_token_id == None or end_token_id == None:
                                            continue
                                        persona_ner_labels[i][start_token_id] = "B"
                                        persona_ner_labels[i][start_token_id + 1:end_token_id + 1] = ["I"] * (
                                                    end_token_id - start_token_id)

                        ############################# knowledge NER ############################# knowledge NER

                        knowledge_can_enc = tokenizer(knowledge_sent, add_special_tokens=False)
                        knowledge_ner_labels = ["O"] * len(knowledge_can_enc['input_ids'])

                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            # print("NER_LABEL: ", ner_label)
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                keyword = knowledge_sent[start:end]
                                start_token_id = knowledge_can_enc.char_to_token(start)
                                end_token_id = knowledge_can_enc.char_to_token(end - 1)
                                if start_token_id == None or end_token_id == None:
                                    continue
                                knowledge_ner_labels[start_token_id] = "B"
                                knowledge_ner_labels[start_token_id + 1:end_token_id + 1] = ["I"] * (
                                            end_token_id - start_token_id)

                        persona_can_enc_new = []
                        persona_ner_labels_enc = []

                        for (can, ner_label) in zip(persona_can_enc, persona_ner_labels):
                            persona_can_enc_new.append(
                                [tokenizer.convert_tokens_to_ids(tokenizer.persona_token)] + can['input_ids'])
                            persona_ner_labels_enc.append([tokenizer.persona_token] + ner_label)

                        knowledge_can_enc = [tokenizer.bos_token_id,
                                             tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)] + \
                                            knowledge_can_enc['input_ids']
                        knowledge_ner_labels = [tokenizer.bos_token, tokenizer.knowledge_token] + knowledge_ner_labels
                        persona_can_enc = list(chain(*persona_can_enc_new))
                        persona_ner_labels_enc = [[ner_label_map[label] for label in sent] for sent in
                                                  persona_ner_labels_enc]
                        persona_ner_labels_enc = list(chain(*persona_ner_labels_enc))
                        knowledge_ner_labels_enc = [ner_label_map[label] for label in knowledge_ner_labels]

                        for i, l in enumerate(persona_ner_labels_enc):
                            if l in [3, 4, 5]:
                                persona_ner_labels_enc[i] = -1
                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3, 4, 5]:
                                knowledge_ner_labels_enc[i] = -1
                        if type(utt["output"]) == list:
                            pred = utt["output"][0]
                        else:
                            pred = utt["output"]
                        dial[-2] = pred

                        dial_enc = [tokenizer(sentence.strip(), add_special_tokens=False)['input_ids'] for sentence in
                                    dial]

                        assert len(persona_can_enc) == len(persona_ner_labels_enc)
                        assert len(knowledge_can_enc) == len(knowledge_ner_labels_enc)

                        dial_new["dialog"] = dial_enc
                        dial_new["persona_grounding"] = persona_ground_enc
                        dial_new["persona_candidates"] = persona_can_enc
                        dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc

                        new_dialogue["utterance"].append(dial_new)
                    persona_enc = persona_can_enc
                    # knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                    new_dialogue["persona"] = persona_enc
                    # new_dialogue["knowledge"] = knowledge_enc
                    new_dialogue["dialogID"] = ID
                    new_dialogue["landmark_link"] = dialogue["landmark_link"]  ##############################
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)
    return all_dataset

def get_dataset_refine_wow(tokenizer, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):
    ner_label_map = {"B":1, "I":2,"O":0, tokenizer.persona_token:3,tokenizer.knowledge_token:4, tokenizer.bos_token:5} ### knowledge_st, persona_st, bos
    ner_label_count = {0:0, 1:0,2:0}
    token_char = tokenizer.convert_ids_to_tokens(5)[0]
    # print(token_char)
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    train_dataset_cache = train_dataset_cache + '_train_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_cache + '_dev_' + type(tokenizer).__name__

    if train_dataset_cache and os.path.isfile(train_dataset_cache):
        print("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        print("Process dataset from %s", train_dataset_path)
        file_train = cached_path(train_dataset_path)
        file_dev = cached_path(dev_dataset_path)
        file_dict = {"valid": file_dev, "train": file_train}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    persona = dialogue["persona"]
                    # knowledge = dialogue["knowledge"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        # print(ID, utt.keys())

                        key = "dialogue" + str(i+1)
                        if key not in utt.keys():
                            continue
                        dial = utt[key]
                        dial_new = dict()
                        persona_can_enc =[]
                        persona_can = dialogue["persona"]

                        knowledge_sent = utt["selected_knowledge"]
                        persona_ner_labels = []
                        persona_ner_labels.append(["O"] * len(persona[0]))

                        persona_can_split = copy.deepcopy(persona_can)

                        ############################# knowledge NER ############################# knowledge NER

                        knowledge_ner_labels = ["O"]*len(knowledge_sent)
                        knowledge_can_split = copy.deepcopy(knowledge_sent)
                        ### 다 뜯어서 태깅하고
                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            # print("NER_LABEL: ", ner_label)
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                keyword = knowledge_sent[start:end]
                                # print(keyword)
                                knowledge_ner_labels[start] = "B"
                                knowledge_ner_labels[start + 1:end] = ["I"] * (end - start - 1)
                        knowledge_can_split = list(knowledge_can_split)
                        assert len(knowledge_can_split) == len(knowledge_ner_labels)

                        for ind, w in enumerate(knowledge_can_split):
                            if w == ' ':
                                del knowledge_can_split[ind]
                                del knowledge_ner_labels[ind]
                        assert len(knowledge_ner_labels) == len(knowledge_can_split)

                        k_sent_words = knowledge_sent.split()
                        k_modi_labels = []
                        tmp_knowledge_can_enc = []
                        char_idx = 0
                        # print(knowledge_ner_labels)
                        # print(knowledge_can_split)
                        # print(len(knowledge_ner_labels))
                        for word in k_sent_words:
                            flag = False
                            diff = 0
                            # print("----------------------------")
                            # 안녕, 하세요
                            correct_syllable_num = len(word)  # 안녕 -> 2
                            # print("word:  ", word, len(word))
                            # print(word)  # 조사에도
                            tokenized_word = tokenizer.tokenize(word)

                            # print(tokenized_word)  # ['조사', '##에도']
                            # print(''.join(tokenized_word))
                            if word != ''.join(tokenized_word):
                                flag = True
                                diff = abs(len(word)-len(''.join(tokenized_word)))
                            contain_unk = True if tokenizer.unk_token in tokenized_word else False
                            for j, token in enumerate(tokenized_word):
                                # print()
                                if not token:
                                    k_modi_labels.append("O")
                                    continue
                                # modi_labels.append(original_clean_labels[char_idx])
                                # print(char_idx)
                                # print(knowledge_ner_labels[char_idx])
                                # print(knowledge_can_split[char_idx])
                                if char_idx >= len(knowledge_ner_labels):
                                    char_idx = len(knowledge_ner_labels)-1
                                # print(char_idx)
                                # print(knowledge_ner_labels[char_idx])
                                # print(knowledge_can_split[char_idx])
                                k_modi_labels.append(knowledge_ner_labels[char_idx])
                                if not contain_unk:
                                    char_idx += len(token)
                            if flag :
                                char_idx -= diff

                            if contain_unk:
                                char_idx += correct_syllable_num
                            tokenized_word[0] = token_char + tokenized_word[0]
                            tmp_knowledge_can_enc.extend(tokenized_word)
                        modi_knowledge_ner_label= [tokenizer.bos_token] + [tokenizer.knowledge_token] +k_modi_labels
                        knowledge_can_enc = [tokenizer.bos_token] + [tokenizer.knowledge_token] + tmp_knowledge_can_enc
                        assert len(modi_knowledge_ner_label)==len(knowledge_can_enc)

                        knowledge_ner_labels_enc = []
                        for tmp in modi_knowledge_ner_label:
                            tmp_ = [ner_label_map[tmp]]
                            knowledge_ner_labels_enc.extend(tmp_)
                        assert len(knowledge_ner_labels_enc) == len(knowledge_can_enc)
                        # print(dial)
                        dial[-2] = utt["output"]

                        dial_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in dial]

                        persona_can_enc = [tokenizer.convert_tokens_to_ids(sentence) for sentence in persona_can_enc]
                        knowledge_can_enc = tokenizer.convert_tokens_to_ids(knowledge_can_enc)

                        # for i, l in enumerate(persona_ner_labels_enc):
                        #     if l in [3,4,5]:
                        #         persona_ner_labels_enc[i] = -1
                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3,4,5]:
                                knowledge_ner_labels_enc[i] = -1
                        # count_dic = Counter(persona_ner_labels_enc)
                        # for l, v in count_dic.items():
                        #     if l != -1:
                        #         ner_label_count[l] += v
                        # count_dic = Counter(knowledge_ner_labels_enc)
                        # for l, v in count_dic.items():
                        #     if l != -1:
                        #         ner_label_count[l] += v


                        dial_new["dialog"] = dial_enc
                        # dial_new["persona_grounding"] = persona_ground_enc
                        dial_new["persona_candidates"] = persona_can_enc
                        # dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc

                        new_dialogue["utterance"].append(dial_new)
                    persona_enc = persona_can_enc
                    # knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                    new_dialogue["persona"] = persona_enc
                    # new_dialogue["knowledge"] = knowledge_enc
                    new_dialogue["dialogID"] = ID
                    # new_dialogue["landmark_link"] = dialogue["landmark_link"]  ##############################
                    dataset_enc[name].append(new_dialogue)


            logger.info("Tokenize and encode the dataset")
            # print("**********############",ner_label_count)
            # breakpoint()
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)

    return all_dataset


def get_dataset_focus_yoonna(tokenizer, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):
    ner_label_map = {"B":1, "I":2,"O":0, tokenizer.persona_token:3,tokenizer.knowledge_token:4, tokenizer.bos_token:5} ### knowledge_st, persona_st, bos

    token_char = tokenizer.convert_ids_to_tokens(5)[0]
    # print(token_char)
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    train_dataset_cache = train_dataset_cache + '_train_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_cache + '_dev_' + type(tokenizer).__name__

    if train_dataset_cache and os.path.isfile(train_dataset_cache):
        print("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        print("Process dataset from %s", train_dataset_path)
        file_train = cached_path(train_dataset_path)
        file_dev = cached_path(dev_dataset_path)
        file_dict = {"train": file_train, "valid": file_dev}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    persona = dialogue["persona"]
                    # knowledge = dialogue["knowledge"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        key = "dialogue" + str(i+1)
                        dial = utt[key]
                        dial_new = dict()
                        persona_can = utt["persona_candidate"]
                        persona_ground = utt["persona_grounding"]
                        knowledge_can = utt["knowledge_candidates"]
                        knowledge_answer = utt["knowledge_answer_index"]
                        knowledge_sent = knowledge_can[knowledge_answer]

                        persona_can_enc = [tokenizer(sentence, add_special_tokens=False) for sentence in persona_can]
                        persona_ground_enc = [1 if item==True else 0 for item in persona_ground]

                        persona_ner_labels = []
                        for i in range(5):
                            persona_ner_labels.append(["O"]*len(persona_can_enc[i]['input_ids']))

                        for ner_label in utt["NER_tagging"].keys(): #LOC, MISC, PER, ORG
                            tmp_persona_index = utt["NER_tagging"][ner_label]["persona_index"] #p_1부터 p_5까지
                            # print(tmp_persona_index)
                            for i in range(5):
                                if tmp_persona_index[f"p_{i+1}"] !=[]: # persona candidate에 태깅된 entity가 있는 경우
                                    for p in range(len(tmp_persona_index[f"p_{i+1}"])):
                                        start, end = tmp_persona_index[f"p_{i+1}"][p]
                                        keyword = persona_can[i][start:end]
                                        start_token_id = persona_can_enc[i].char_to_token(start)
                                        end_token_id = persona_can_enc[i].char_to_token(end-1)
                                        if start_token_id == None or end_token_id == None:
                                            continue
                                        persona_ner_labels[i][start_token_id] = "B"
                                        persona_ner_labels[i][start_token_id+1:end_token_id+1] = ["I"] * (end_token_id - start_token_id)

                        ############################# knowledge NER ############################# knowledge NER

                        knowledge_can_enc = tokenizer(knowledge_sent, add_special_tokens=False)
                        knowledge_ner_labels = ["O"]*len(knowledge_can_enc['input_ids'])

                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            # print("NER_LABEL: ", ner_label)
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                keyword = knowledge_sent[start:end]
                                start_token_id = knowledge_can_enc.char_to_token(start)
                                end_token_id = knowledge_can_enc.char_to_token(end-1)
                                if start_token_id == None or end_token_id == None:
                                    continue
                                knowledge_ner_labels[start_token_id] = "B"
                                knowledge_ner_labels[start_token_id+1:end_token_id+1] = ["I"] * (end_token_id - start_token_id)

                        persona_can_enc_new = []
                        persona_ner_labels_enc = []
                        for (can, ner_label) in zip(persona_can_enc, persona_ner_labels):
                            persona_can_enc_new.append([tokenizer.convert_tokens_to_ids(tokenizer.persona_token)] + can['input_ids'])
                            persona_ner_labels_enc.append([tokenizer.persona_token] + ner_label)

                        knowledge_can_enc = [tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)] + knowledge_can_enc['input_ids']
                        knowledge_ner_labels = [tokenizer.bos_token, tokenizer.knowledge_token] + knowledge_ner_labels
                        persona_can_enc = list(chain(*persona_can_enc_new))
                        persona_ner_labels_enc = [[ner_label_map[label] for label in sent] for sent in persona_ner_labels_enc]
                        persona_ner_labels_enc = list(chain(*persona_ner_labels_enc))
                        knowledge_ner_labels_enc = [ner_label_map[label] for label in knowledge_ner_labels]

                        for i, l in enumerate(persona_ner_labels_enc):
                            if l in [3,4,5]:
                                persona_ner_labels_enc[i] = -1
                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3,4,5]:
                                knowledge_ner_labels_enc[i] = -1

                        dial_enc = [tokenizer(sentence.strip(), add_special_tokens=False)['input_ids'] for sentence in dial]

                        assert len(persona_can_enc) == len(persona_ner_labels_enc)
                        assert len(knowledge_can_enc) == len(knowledge_ner_labels_enc)

                        dial_new["dialog"] = dial_enc
                        dial_new["persona_grounding"] = persona_ground_enc
                        dial_new["persona_candidates"] = persona_can_enc
                        dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc

                        new_dialogue["utterance"].append(dial_new)
                    persona_enc = persona_can_enc
                    # knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                    new_dialogue["persona"] = persona_enc
                    # new_dialogue["knowledge"] = knowledge_enc
                    new_dialogue["dialogID"] = ID
                    new_dialogue["landmark_link"] = dialogue["landmark_link"]  ##############################
                    dataset_enc[name].append(new_dialogue)


            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)
    return all_dataset
