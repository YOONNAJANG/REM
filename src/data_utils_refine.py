import logging
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from tqdm import tqdm
from transformers import cached_path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sklearn.utils import shuffle
import json


special_tokens_focus = {'machine_token':'<machine>', 'human_token':'<human>', 'persona_token':'<persona>', 'knowledge_token':'<knowledge>'}

MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "ner_labels"]
logger = logging.getLogger(__file__)

def add_special_tokens_(model, tokenizer, special_tokens):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # e.g., special_tokens = {'subj_token': '<subj>'}
    # orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
    tokenizer.__dict__.update(special_tokens)
    if num_added_tokens > 0:
        print(num_added_tokens, 'tokens are added, total ', len(tokenizer), 'tokens in vocab.')
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

def pad_dataset_data(dataset, padding):
    max_l = max(len(x) for x in dataset["input_ids"])
    # max_ner_l = max(len(x) for x in dataset["ner_labels"])
    max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
    dataset['input_ids'] = [x + [padding] * (max_l - len(x)) for x in dataset['input_ids']]
    dataset['ner_labels'] = [x + [-1] * (max_l - len(x)) for x in dataset['ner_labels']]
    dataset['decoder_input_ids'] = [x + [padding] * ((max_dec_l) - len(x)) for x in dataset['decoder_input_ids']]
    dataset['lm_labels'] = [x + [-100] * (max_dec_l - len(x)) for x in dataset['lm_labels']]
    return dataset


def build_input_data(args, data_name, tokenizer, history, persona_cans, persona_ner_label, golden_knowledge, knowledge_ner_label): #gk|p|history|u' -> u
    if 'bart' in tokenizer.name_or_path:
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        dec_bos = 2  # tokenizer.decoder_start_token_id
        gold_knowledge = golden_knowledge
    else:
        bos, dec_bos, eos = tokenizer.eos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id
        gold_knowledge = golden_knowledge[1:]

    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)

    history_, reply = history[:-1], history[-1]
    knowledge_label = []
    knowledge_label.extend(knowledge_ner_label)
    history_ = [human_st] + history_[0]
    if data_name == "focus":
        enc_sequence = gold_knowledge + persona_cans
        enc_sequence.extend(history_)
        ner_label = knowledge_label + persona_ner_label
    else :
        enc_sequence = gold_knowledge
        enc_sequence.extend(history_)
        ner_label = knowledge_label
    dec_sequence = [dec_bos] + reply + [eos]

    instance = dict()

    if 'bart' in tokenizer.name_or_path:
        instance['input_ids'] = enc_sequence
        instance['ner_labels'] = ner_label
    else:
        instance['input_ids'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Rewrite the utterance considering the knowledge: ')) + enc_sequence[1:]
        instance['ner_labels'] = len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Rewrite the utterance considering the knowledge: '))) * [-1] + ner_label[1:]

    instance['decoder_input_ids'] = dec_sequence[:-1]
    instance['lm_labels'] = dec_sequence[1:]
    return instance

def dataloader_train(args, data_name, tokenizer, train_path, train_cache_path, dev_path, dev_cache_path, multi=False):

    regen_data = get_dataset_refine_data(tokenizer, data_name, train_dataset_path=train_path,
                                    train_dataset_cache=train_cache_path,
                                    dev_dataset_path=dev_path,
                                    dev_dataset_cache=dev_cache_path)

    print("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for (key, value) in regen_data.items():
        for data in tqdm(value):
            utterance = data['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2 * args.max_history):]
                # if len(history[-1]) > 256:
                #     continue
                if data_name == "focus":

                    persona_cans = utt['persona_candidates']
                    persona_ner_label = utt['persona_ner_label']
                else:
                    persona_cans = None
                    persona_ner_label = None
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_data(args, data_name, tokenizer, history, persona_cans, persona_ner_label,
                                             golden_knowledge,
                                             knowledge_ner_label)
                for input_name, input_array in instance.items():
                    datasets[key][input_name].append(input_array)
    if multi:
        return datasets

    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_data(dataset, padding=tokenizer.pad_token_id)

        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    print("Build train and valid dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    return train_dataset, valid_dataset

def dataloader_multi(args, tokenizer, data_list):
    valid_data_type = data_list[0]
    print("Pad inputs and convert to Tensor")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    key_list = ['input_ids', 'decoder_input_ids', 'lm_labels', 'ner_labels']
    for data_type in data_list:
        tmp_dataset = dataloader_train(args, data_type, tokenizer, multi=True)

        for key in key_list :
            for instance in tmp_dataset["train"][key]:
                datasets["train"][key].append(instance)

        if data_type == valid_data_type:
            for key in key_list:
                for instance in tmp_dataset["valid"][key]:
                    datasets["valid"][key].append(instance)

    input_id, decoder_input_id, lm_labels, ner_labels = shuffle(datasets["train"]["input_ids"], datasets["train"]["decoder_input_ids"], datasets["train"]["lm_labels"],datasets["train"]["ner_labels"])
    datasets["train"]["input_ids"] = input_id
    datasets["train"]["decoder_input_ids"] = decoder_input_id
    datasets["train"]["lm_labels"] = lm_labels
    datasets["train"]["ner_labels"] = ner_labels


    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_data(dataset, padding=tokenizer.pad_token_id)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    print("Build train and valid dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    return train_dataset, valid_dataset

def get_dataset_refine_data(tokenizer, data_name, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):
    ner_label_map = {"B": 1, "I": 2, "O": 0, tokenizer.persona_token: 3, tokenizer.knowledge_token: 4,
                     tokenizer.bos_token: 5}

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
                if data_name =="cmodog":
                    dataset = dataset["data"]
                dataset_enc = dict()
                dataset_enc[name] = list()

                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        if data_name in ["wow", "cmudog"]:
                            key = "dialogue" + str(i + 1)
                        else:
                            key = "dialogue" + str(i)
                        if key not in utt.keys():
                            continue
                        dial = utt[key]
                        dial_new = dict()
                        if data_name == "focus":

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
                                tmp_persona_index = utt["NER_tagging"][ner_label]["persona_index"]

                                for i in range(5):
                                    if tmp_persona_index[f"p_{i + 1}"] != []:
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
                        elif data_name in ["wow", "cmudog"]:
                            knowledge_sent = utt["selected_knowledge"]

                        ### knowledge NER
                        knowledge_can_enc = tokenizer(knowledge_sent, add_special_tokens=False)
                        knowledge_ner_labels = ["O"] * len(knowledge_can_enc['input_ids'])

                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                start_token_id = knowledge_can_enc.char_to_token(start)
                                end_token_id = knowledge_can_enc.char_to_token(end - 1)
                                if start_token_id == None or end_token_id == None:
                                    continue
                                knowledge_ner_labels[start_token_id] = "B"
                                knowledge_ner_labels[start_token_id + 1:end_token_id + 1] = ["I"] * (
                                            end_token_id - start_token_id)

                        if data_name == "focus":
                            persona_can_enc_new = []
                            persona_ner_labels_enc = []

                            for (can, ner_label) in zip(persona_can_enc, persona_ner_labels):
                                persona_can_enc_new.append(
                                    [tokenizer.convert_tokens_to_ids(tokenizer.persona_token)] + can['input_ids'])
                                persona_ner_labels_enc.append([tokenizer.persona_token] + ner_label)

                            persona_can_enc = list(chain(*persona_can_enc_new))
                            persona_ner_labels_enc = [[ner_label_map[label] for label in sent] for sent in
                                                      persona_ner_labels_enc]
                            persona_ner_labels_enc = list(chain(*persona_ner_labels_enc))
                            for i, l in enumerate(persona_ner_labels_enc):
                                if l in [3, 4, 5]:
                                    persona_ner_labels_enc[i] = -1
                            assert len(persona_can_enc) == len(persona_ner_labels_enc)


                        knowledge_can_enc = [tokenizer.bos_token_id,
                                             tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)] + \
                                            knowledge_can_enc['input_ids']
                        knowledge_ner_labels = [tokenizer.bos_token, tokenizer.knowledge_token] + knowledge_ner_labels

                        knowledge_ner_labels_enc = [ner_label_map[label] for label in knowledge_ner_labels]


                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3, 4, 5]:
                                knowledge_ner_labels_enc[i] = -1
                        assert len(knowledge_can_enc) == len(knowledge_ner_labels_enc)

                        if type(utt["output"]) == list:
                            pred = utt["output"][0]
                        else:
                            pred = utt["output"]

                        dial[-2] = pred

                        dial_enc = [tokenizer(sentence.strip(), add_special_tokens=False)['input_ids'] for sentence in
                                    dial]

                        dial_new["dialog"] = dial_enc
                        if data_name == "focus":
                            dial_new["persona_grounding"] = persona_ground_enc
                            dial_new["persona_candidates"] = persona_can_enc
                            dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc
                        new_dialogue["utterance"].append(dial_new)
                    if data_name == "focus":
                        persona_enc = persona_can_enc
                        new_dialogue["persona"] = persona_enc
                        new_dialogue["landmark_link"] = dialogue["landmark_link"]  ##############################

                    new_dialogue["dialogID"] = ID
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)
    return all_dataset

def dataloader_test(args, data_name, tokenizer,test_dataset_path, test_dataset_cache, multi=False):

    regen_data = get_dataset_refine_data_test(tokenizer, data_name, test_dataset_path=test_dataset_path,
                                    test_dataset_cache=test_dataset_cache)


    print("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    for (key, value) in regen_data.items():
        for data in tqdm(value):
            utterance = data['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2 * args.max_history):]
                if data_name == "focus":
                    persona_cans = utt['persona_candidates']
                    persona_ner_label = utt['persona_ner_label']
                else:
                    persona_cans = None
                    persona_ner_label = None
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_data(args,data_name, tokenizer, history, persona_cans, persona_ner_label,
                                             golden_knowledge,
                                             knowledge_ner_label)
                for input_name, input_array in instance.items():
                    datasets[key][input_name].append(input_array)
    if multi:
        return datasets
    print("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}
    # print(datasets)
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset_data(dataset, padding=tokenizer.pad_token_id)
        # print(dataset)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    print("Build test dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])
    return test_dataset

def get_dataset_refine_data_test(tokenizer, data_name, test_dataset_path, test_dataset_cache):
    ner_label_map = {"B":1, "I":2,"O":0, tokenizer.persona_token:3,tokenizer.knowledge_token:4, tokenizer.bos_token:5} ### knowledge_st, persona_st, bos

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
                if data_name == "cmudog":
                    dataset = dataset["data"]
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset:
                    ID = dialogue["dialogID"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        if data_name in ["wow", "cmudog", "chatgpt"]:
                            key = "dialogue" + str(i + 1)
                        else:
                            key = "dialogue" + str(i + 1)
                        dial = utt[key]
                        dial_new = dict()

                        if data_name == "focus":
                            persona_can = utt["persona_candidate"]
                            persona_ground = utt["persona_grounding"]
                            knowledge_can = utt["knowledge_candidates"]
                            knowledge_answer = utt["knowledge_answer_index"]
                            knowledge_sent = knowledge_can[knowledge_answer]
                            persona_can_enc = [tokenizer(sentence, add_special_tokens=False) for sentence in
                                               persona_can]
                            persona_ground_enc = [1 if item == True else 0 for item in persona_ground]
                            persona_ner_labels = []
                            for i in range(5):
                                persona_ner_labels.append(["O"] * len(persona_can_enc[i]['input_ids']))

                            for ner_label in utt["NER_tagging"].keys():  # LOC, MISC, PER, ORG
                                tmp_persona_index = utt["NER_tagging"][ner_label]["persona_index"]  # p_1부터 p_5까지
                                for i in range(5):
                                    if tmp_persona_index[f"p_{i + 1}"] != []:  # persona candidate에 태깅된 entity가 있는 경우
                                        for p in range(len(tmp_persona_index[f"p_{i + 1}"])):
                                            start, end = tmp_persona_index[f"p_{i + 1}"][p]

                                            start_token_id = persona_can_enc[i].char_to_token(start)
                                            end_token_id = persona_can_enc[i].char_to_token(end - 1)
                                            if start_token_id == None or end_token_id == None:
                                                continue
                                            persona_ner_labels[i][start_token_id] = "B"
                                            persona_ner_labels[i][start_token_id + 1:end_token_id + 1] = ["I"] * (
                                                    end_token_id - start_token_id)

                        else:
                            knowledge_sent = utt["selected_knowledge"]

                        ##### knowledge NER
                        knowledge_can_enc = tokenizer(knowledge_sent, add_special_tokens=False)
                        knowledge_ner_labels = ["O"] * len(knowledge_can_enc['input_ids'])

                        for ner_label in utt["NER_tagging"].keys():
                            tmp_knowledge_index = utt["NER_tagging"][ner_label]["knowledge_index"]
                            for k in range(len(tmp_knowledge_index)):
                                start, end = tmp_knowledge_index[k]
                                start_token_id = knowledge_can_enc.char_to_token(start)
                                end_token_id = knowledge_can_enc.char_to_token(end - 1)
                                if start_token_id == None or end_token_id == None:
                                    continue
                                knowledge_ner_labels[start_token_id] = "B"
                                knowledge_ner_labels[start_token_id + 1:end_token_id + 1] = ["I"] * (
                                            end_token_id - start_token_id)

                        if data_name == "focus":
                            persona_can_enc_new = []
                            persona_ner_labels_enc = []
                            for (can, ner_label) in zip(persona_can_enc, persona_ner_labels):
                                persona_can_enc_new.append(
                                    [tokenizer.convert_tokens_to_ids(tokenizer.persona_token)] + can['input_ids'])
                                persona_ner_labels_enc.append([tokenizer.persona_token] + ner_label)
                            persona_can_enc = list(chain(*persona_can_enc_new))
                            persona_ner_labels_enc = [[ner_label_map[label] for label in sent] for sent in
                                                      persona_ner_labels_enc]
                            persona_ner_labels_enc = list(chain(*persona_ner_labels_enc))
                            for i, l in enumerate(persona_ner_labels_enc):
                                if l in [3, 4, 5]:
                                    persona_ner_labels_enc[i] = -1
                            assert len(persona_can_enc) == len(persona_ner_labels_enc)

                        knowledge_can_enc = [tokenizer.bos_token_id,
                                             tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)] + \
                                            knowledge_can_enc['input_ids']
                        knowledge_ner_labels = [tokenizer.bos_token, tokenizer.knowledge_token] + knowledge_ner_labels
                        knowledge_ner_labels_enc = [ner_label_map[label] for label in knowledge_ner_labels]


                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3, 4, 5]:
                                knowledge_ner_labels_enc[i] = -1
                        if type(utt["output"]) == list:
                            pred = utt["output"][0]
                        else:
                            pred = utt["output"]
                        dial[-2] = pred
                        dial_enc = [tokenizer(sentence.strip(), add_special_tokens=False)['input_ids'] for sentence in dial]
                        assert len(knowledge_can_enc) == len(knowledge_ner_labels_enc)

                        dial_new["dialog"] = dial_enc
                        if data_name == "focus":
                            dial_new["persona_grounding"] = persona_ground_enc
                            dial_new["persona_candidates"] = persona_can_enc
                            dial_new["persona_ner_label"] = persona_ner_labels_enc
                        dial_new["golden_knowledge"] = knowledge_can_enc
                        dial_new["knowledge_ner_label"] = knowledge_ner_labels_enc
                        new_dialogue["utterance"].append(dial_new)
                    new_dialogue["dialogID"] = ID

                    if data_name == "focus":
                        persona_enc = persona_can_enc
                        new_dialogue["persona"] = persona_enc
                        new_dialogue["landmark_link"] = dialogue["landmark_link"]
                    dataset_enc[name].append(new_dialogue)

            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            torch.save(dataset, test_dataset_cache)

    return all_dataset