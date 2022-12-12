import logging
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import copy
from collections import Counter
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
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        congen_model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return tokenizer, model, congen_model



def build_input_focus(tokenizer, history, persona_cans, persona_ner_label, golden_knowledge, knowledge_ner_label): #gk|p|history|u' -> u
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

    assert len(persona_ner_label) == len(persona_cans)

    ner_label = [bos] + knowledge_label + persona_ner_label
    # print(enc_sequence)
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

    train_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/our_train.json"

    train_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/our_train_cache.tar.gz"
    dev_dataset_path = "/home/data/ssh5131/focus_modeling/for_refiner_v2/our_dev.json"
    dev_dataset_cache = "/home/data/ssh5131/focus_modeling/for_refiner_v2/our_dev_cache.tar.gz"

    regen_data = get_dataset_refine(tokenizer, train_dataset_path=train_dataset_path,
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
                persona_ner_label = utt['persona_ner_label']
                golden_knowledge = utt['golden_knowledge']
                knowledge_ner_label = utt['knowledge_ner_label']
                instance = build_input_focus(tokenizer, history, persona_cans, persona_ner_label, golden_knowledge,
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


def get_dataset_refine(tokenizer, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):
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
                        persona_can_enc =[]
                        persona_can = utt["persona_candidate"]
                        if len(persona_can) > 5:
                            persona_can = persona_can[:5]
                        persona_ground = utt["persona_grounding"]
                        if len(persona_ground) > 5:
                            persona_ground = persona_ground[:5]
                        knowledge_can = utt["knowledge_candidates"]
                        knowledge_answer = utt["knowledge_answer_index"]
                        knowledge_sent = knowledge_can[knowledge_answer]
                        persona_ner_labels = []
                        for i in range(5):
                            persona_ner_labels.append(["O"]*len(persona[i]))
                        persona_can_split = copy.deepcopy(persona_can)
                        ### 다 뜯어서 태깅하고
                        for ner_label in utt["NER_tagging"].keys():
                            tmp_persona_index =  utt["NER_tagging"][ner_label]["persona_index"]
                            for i in range(5):
                                if tmp_persona_index[f"p_{i+1}"] !=[]:
                                    for p in range(len(tmp_persona_index[f"p_{i+1}"])):
                                        # persona_ner_labels[i].append(tmp_persona_index[f"p_{i+1}"][p])
                                        start, end = tmp_persona_index[f"p_{i+1}"][p]
                                        keyword = persona_can[i][start:end]
                                        persona_ner_labels[i][start] = "B"
                                        persona_ner_labels[i][start+1:end] = ["I"] * (end-start-1)
                                persona_can_split[i] = list(persona_can_split[i] )
                                assert len(persona_can_split[i]) == len(persona_ner_labels[i])
                        ### 공백, 거기에 맞는 태깅 제거
                        for i in range(5):
                            for ind, w in enumerate(persona_can_split[i]):
                                if w == ' ':
                                    del persona_can_split[i][ind]
                                    del persona_ner_labels[i][ind]
                            assert len(persona_ner_labels[i]) == len(list(persona_can_split[i]))
                        ### 늘어나는 토큰만큼 레이블추가
                        modi_persona_ner_labels = []
                        for i in range(5):
                            sent_words = persona_can[i].split()
                            modi_labels =[]
                            tmp_persona_can_enc =[]
                            char_idx=0
                            for word in sent_words:
                                flag = False
                                diff = 0
                                # 안녕, 하세요
                                correct_syllable_num = len(word)  # 안녕 -> 2
                                # print("word:  ", word, len(word))
                                tokenized_word = tokenizer.tokenize(word)
                                contain_unk = True if tokenizer.unk_token in tokenized_word else False
                                for j, token in enumerate(tokenized_word):
                                    if not token:
                                        modi_labels.append("O")
                                        continue
                                    # modi_labels.append(original_clean_labels[char_idx])
                                    if char_idx >= len(persona_ner_labels[i]):
                                        char_idx = len(persona_ner_labels[i]) - 1

                                    modi_labels.append(persona_ner_labels[i][char_idx])
                                    if not contain_unk:
                                        char_idx += len(token)
                                if flag:
                                    char_idx -= diff
                                if contain_unk:
                                    char_idx += correct_syllable_num

                                tokenized_word[0] = token_char+tokenized_word[0]
                                tmp_persona_can_enc.extend(tokenized_word)
                            modi_persona_ner_labels.extend(modi_labels)
                            persona_can_enc.extend(tmp_persona_can_enc)

                        modi_persona_ner_labels =[tokenizer.persona_token] + modi_persona_ner_labels
                        persona_can_enc = [tokenizer.persona_token] + persona_can_enc
                        assert len(modi_persona_ner_labels) == len(persona_can_enc)

                        persona_ground_enc = [1 if item==True else 0 for item in persona_ground]
                        persona_ner_labels_enc = []
                        # print(modi_persona_ner_labels)
                        # print(persona_can_enc)

                        for tmp in modi_persona_ner_labels:
                            tmp_ = ner_label_map[tmp]
                            persona_ner_labels_enc.append(tmp_)
                        # print(persona_ner_labels_enc)
                        assert len(persona_ner_labels_enc) == len(persona_can_enc)
                        # for l, p in zip(persona_ner_labels_enc, persona_can_enc):
                        #     assert  len(l) == len(p)
                        ############################# knowledge NER ############################# knowledge NER
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

                        ### 공백, 거기에 맞는 태깅 제거

                        for ind, w in enumerate(knowledge_can_split):
                            if w == ' ':
                                del knowledge_can_split[ind]
                                del knowledge_ner_labels[ind]
                        assert len(knowledge_ner_labels) == len(knowledge_can_split)
                        ### 늘어나는 토큰만큼 레이블추가


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
                        # print("---------------")
                        # print(modi_knowledge_ner_label)
                        # print(knowledge_can_enc)
                        # print(ner_label_map)
                        for tmp in modi_knowledge_ner_label:
                            tmp_ = [ner_label_map[tmp]]
                            knowledge_ner_labels_enc.extend(tmp_)
                        assert len(knowledge_ner_labels_enc) == len(knowledge_can_enc)

                        dial_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in dial]
                        # knowledge_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge_can]
                        # persona_can_enc_ = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona_can]
                        persona_can_enc = [tokenizer.convert_tokens_to_ids(sentence) for sentence in persona_can_enc]
                        knowledge_can_enc = tokenizer.convert_tokens_to_ids(knowledge_can_enc)
                        # print(knowledge_can_enc)
                        ######special token 없애기###########

                        for i, l in enumerate(persona_ner_labels_enc):
                            if l in [3,4,5]:
                                persona_ner_labels_enc[i] = -1
                        for i, l in enumerate(knowledge_ner_labels_enc):
                            if l in [3,4,5]:
                                knowledge_ner_labels_enc[i] = -1
                        count_dic = Counter(persona_ner_labels_enc)
                        for l, v in count_dic.items():
                            if l != -1:
                                ner_label_count[l] += v
                        count_dic = Counter(knowledge_ner_labels_enc)
                        for l, v in count_dic.items():
                            if l != -1:
                                ner_label_count[l] += v


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
            print("**********############",ner_label_count)
            # breakpoint()
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)

    return all_dataset

