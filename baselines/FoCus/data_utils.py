#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader, TensorDataset
from choose_knowledge import *
from utils_focus import get_dataset_only_train_dev, get_dataset_only_test
import json
import random
from random import randrange

special_tokens = {'machine_token':'<machine>', 'human_token':'<human>', 'persona_token':'<persona>', 'knowledge_token':'<knowledge>'}

MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels",
                "persona_candidates", "persona_can_idx", "persona_grounding",
                "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding", "reply"]

CTXT_MODEL_INPUTS = ["input_ids", "input_eos", "decoder_input_ids", "lm_labels",
                     "persona_candidates", "persona_can_idx", "persona_grounding",
                     "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding",
                     "tot_knowledge", "tot_knowledge_eos", "reply", "dialog", "history_list"]

PADDED_INPUTS = ["decoder_input_ids", "lm_labels"]
logger = logging.getLogger(__file__)

num_persona = 5
num_knowledge = 10


def pad_dataset(dataset, padding=1):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_l = max(len(x) for x in dataset["decoder_input_ids"])
    max_l_reply = max(len(x) for x in dataset["reply"])

    ###############to delete examples with too long knowledge candidates######################
    remove_list = list()
    persona_nonlist = list()
    for idx_1, x in enumerate(dataset["knowledge_candidates"]):
        for idx_2, i in enumerate(x):
            if len(i) > 500 or type(i) != list:
                print("knowledge", len(i), type(i))
                remove_list.append(idx_1)

    for idx_1, x in enumerate(dataset["persona_candidates"]):
        if len(x) != num_persona or type(x) != list:
            remove_list.append(idx_1)
            persona_nonlist.append(idx_1)
        for idx_2, i in enumerate(x):
            if len(i) > 500 or type(i) != list:
                remove_list.append(idx_1)

    for idx_1, x in enumerate(dataset["tot_knowledge"]):
        for idx_2, i in enumerate(x):
            if type(i) != list:
                remove_list.append(idx_1)
            elif len(i) > 200:
                dataset["tot_knowledge"][idx_1][idx_2] = i[:200]

    remove_list = list(set(remove_list))
    print("remove list: ", len(remove_list))

    if len(remove_list) != 0:
        new_dataset = defaultdict(list)
        for input in MODEL_INPUTS:
            for i, element in enumerate(dataset[input]):
                if i in remove_list:
                    continue
                else:
                    if input == 'persona_candidates':
                        assert len(element) == num_persona
                    new_dataset[input].append(element)
    else:
        new_dataset = dataset

    max_l_knowledge_cans = max([len(i) for x in new_dataset["knowledge_candidates"] for i in x])
    max_l_tot_knowledge = max([len(i) for x in new_dataset["tot_knowledge"] for i in x])
    max_l_persona_cans = max([len(i) for x in new_dataset["persona_candidates"] for i in x])
    max_l_history = max([len(i) for x in new_dataset["history_list"] for i in x])
    max_l_dialog = max(len(x) for x in new_dataset["dialog"])

    for name in PADDED_INPUTS:
        new_dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in new_dataset[name]]
    #print('new dataset : ', new_dataset["input_ids"])

    new_dataset["input_ids"] = [x + [padding] * (max_enc_l - len(x)) for x in new_dataset["input_ids"]]
    new_dataset["reply"] = [x + [padding] * (max_l_reply - len(x)) for x in new_dataset["reply"]]

    knowledge_list = list()
    for i, knowledges in enumerate(new_dataset["knowledge_candidates"]):
        candidates_list = list()
        for candidates in knowledges:
            padded_candidate = candidates + [padding] * (max_l_knowledge_cans - len(candidates))
            candidates_list.append(padded_candidate)
        knowledge_list.append(candidates_list)
    new_dataset["knowledge_candidates"] = knowledge_list

    persona_list = list()
    for i, personas in enumerate(new_dataset["persona_candidates"]):
        candidates_list = list()
        for candidates in personas:
            padded_candidate = candidates + [padding] * (max_l_persona_cans - len(candidates))
            candidates_list.append(padded_candidate)
        persona_list.append(candidates_list)
    new_dataset["persona_candidates"] = persona_list

    tot_knowledge_list = list()
    for i, tot_kn in enumerate(new_dataset["tot_knowledge"]):
        candidates_list = list()
        for candidates in tot_kn:
            padded_candidate = candidates + [padding] * (max_l_tot_knowledge - len(candidates))
            candidates_list.append(padded_candidate)
        tot_knowledge_list.append(candidates_list)
    new_dataset["tot_knowledge"] = tot_knowledge_list

    history_list = list()
    for i, history in enumerate(new_dataset["history_list"]):
        candidate_list = list()
        for candidate in history:
            padded_candidate = candidate + [padding] * (max_l_history - len(candidate))
            candidate_list.append(padded_candidate)
        history_list.append(candidate_list)
    new_dataset["history_list"] = history_list

    new_dataset["dialog"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog"]]
    new_dataset["dialog_tti"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog_tti"]]
    return new_dataset

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


def build_input_for_bart(args, persona, knowledge, landmark_link, history, persona_cans, persona_grounding, knowledge_cans, knowledge_answer_idx, ID, tokenizer, retrieval_type, sentence_knowledge_vector_dict=None, BM25Okapi = None,  table=None, inference=False):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos = tokenizer.bos_token_id
    dec_bos = 2 #tokenizer.decoder_start_token_id
    eos = tokenizer.eos_token_id
    machine_st = tokenizer.convert_tokens_to_ids(tokenizer.machine_token)
    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
    persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)
    #machine: 50265 human: 50266 persona: 50267 knowledge: 50268 padding: 1 bos: 0 eos: 2

    history, reply = history[:-1], history[-1]
    history_list = history
    history_context, history_question = history_list[:-1], history_list[-1]
    if len(history_context) == 0:
        history_context = [1]
    else:
        history_context = list(chain(*history_context))

    history_list_new = []
    history_list_new.append(history_context)
    history_list_new.append(history_question)


    history = [[human_st if i % 2 == 0 else machine_st] + s for i, s in enumerate(history)]

    knowledge = [para for para in knowledge if len(para) > 0]
    if inference == False:
        if len(knowledge) > 1:
            if retrieval_type == "DPR":
                if args.DPR_train is False:
                    chosen_knowledge = choose_knowledge_dpr(landmark_link, history[-1], tokenizer=tokenizer, table=table)
                elif args.DPR_train is True:
                    chosen_knowledge = choose_knowledge_trained_dpr(landmark_link, history[-1], sentence_knowledge_vector_dict, model_key='bart', tokenizer=tokenizer, table=table)
            elif retrieval_type == "TFIDF":
                chosen_knowledge = choose_knowledge_with_tfidf(knowledge, history[-1], table)
            elif retrieval_type == "TFIDF_sen":
                chosen_knowledge = choose_knowledge_sentence_with_tfidf(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', table=table)
            elif retrieval_type == "BM25":
                chosen_knowledge = choose_knowledge_with_bm25(knowledge, history[-1], BM25Okapi)
            elif retrieval_type == "BM25_sen":
                chosen_knowledge = choose_knowledge_sentence_with_bm25(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', BM25Okapi=BM25Okapi)

        else:
            chosen_knowledge = knowledge[0:5]
    else:
        chosen_knowledge = knowledge_cans[knowledge_answer_idx]

    paragraphs = []
    for para in chosen_knowledge:
        #for para in knowledge:
        if len(para) > 100:
            short_para = para[:100]
        else:
            short_para = para
        paragraphs.append(short_para)


    landmark_name_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(landmark_link.split('/')[-1]))



    if len(history) == 1:
        #enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history
        enc_sequence = [[bos]] + [landmark_name_tokenized] + [[persona_st] + list(chain(*persona))] + history
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = [[bos]] + history

    else:
        #enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
        enc_sequence = [[bos]] + [landmark_name_tokenized] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = [[bos]] + [list(chain(*history))]


    instance = {}
    instance["input_ids"] = list(chain(*enc_sequence))
    #instance["input_eos"] = len(list(chain(*enc_sequence)))-1
    instance["input_eos"] = 0
    instance["history_list"] = history_list_new
    instance["dialog"] = list(chain(*dialog))
    instance["decoder_input_ids"] = dec_sequence[:-1]
    instance["lm_labels"] = dec_sequence[1:]
    instance["persona_candidates"] = [[bos] + [persona_st] + can + [eos] for can in persona_cans]
    #instance["persona_can_idx"] = [len(can)-1 for can in instance["persona_candidates"]]
    instance["persona_can_idx"] = [0 for _ in instance["persona_candidates"]]
    instance["persona_grounding"] = persona_grounding
    #landmark name added to the candidates!
    #instance["knowledge_candidates"] = [[bos] + [knowledge_st] + can[:100] + [eos] if len(can) > 100 else [bos] + [knowledge_st] + can + [eos] for can in knowledge_cans]
    instance["knowledge_candidates"] = [[bos] + [knowledge_st] + landmark_name_tokenized + can[:100] + [eos] if len(can) > 100 else [bos] + [knowledge_st] + landmark_name_tokenized + can + [eos] for can in knowledge_cans]
    #instance["knowledge_can_idx"] = [len(can)-1 for can in instance["knowledge_candidates"]]
    instance["knowledge_can_idx"] = [0 for _ in instance["knowledge_candidates"]]
    instance["knowledge_grounding"] = knowledge_answer_idx
    instance["mc_token_ids"] = 0
    instance["dialog_ID"] = ID
    instance["reply"] = reply
    instance['tot_knowledge'] = paragraphs
    #instance['tot_knowledge_eos'] = [len(p)-1 for p in paragraphs]
    instance['tot_knowledge_eos'] = [0 for _ in paragraphs]

    assert len(instance["decoder_input_ids"]) == len(instance["lm_labels"])

    return instance




def build_input_for_t5(args, persona, knowledge, landmark_link, history, persona_cans, persona_grounding, knowledge_cans, knowledge_answer_idx, ID, tokenizer, retrieval_type, sentence_knowledge_vector_dict=None, BM25Okapi = None,  table=None, inference=False):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    eos = tokenizer.eos_token_id
    dec_bos = 0 #tokenizer.decoder_start_token_id
    machine_st = tokenizer.convert_tokens_to_ids(tokenizer.machine_token)
    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
    persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)
    #machine: 32100 human: 32101 persona: 32101 knowledge: 32102 padding: 0 eos: 1

    history, reply = history[:-1], history[-1]
    history_list = history

    history = [[human_st if i % 2 == 0 else machine_st] + s for i, s in enumerate(history)]

    knowledge = [para for para in knowledge if len(para) > 0]

    if inference == False:
        if len(knowledge) > 1:
            if retrieval_type == "DPR":
                if args.DPR_train is False:
                    chosen_knowledge = choose_knowledge_dpr(landmark_link, history[-1], tokenizer=tokenizer, table=table)
                elif args.DPR_train is True:
                    chosen_knowledge = choose_knowledge_trained_dpr(landmark_link, history[-1], sentence_knowledge_vector_dict, model_key='bart', tokenizer=tokenizer, table=table)
            elif retrieval_type == "TFIDF":
                chosen_knowledge = choose_knowledge_with_tfidf(knowledge, history[-1], table)
            elif retrieval_type == "TFIDF_sen":
                chosen_knowledge = choose_knowledge_sentence_with_tfidf(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', table=table)
            elif retrieval_type == "BM25":
                chosen_knowledge = choose_knowledge_with_bm25(knowledge, history[-1], BM25Okapi)
            elif retrieval_type == "BM25_sen":
                chosen_knowledge = choose_knowledge_sentence_with_bm25(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', BM25Okapi=BM25Okapi)

        else:
            chosen_knowledge = knowledge[0:5]
    else:
        chosen_knowledge = knowledge_cans[knowledge_answer_idx]

    paragraphs = []
    for para in chosen_knowledge:
        if len(para) > 100:
            short_para = para[:100]
        else:
            short_para = para
        paragraphs.append(short_para)

    landmark_name_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(landmark_link.split('/')[-1]))


    if len(history) == 1:
        #enc_sequence = [[persona_st] + list(chain(*persona))] + history + [[eos]]
        enc_sequence = [[persona_st] + landmark_name_tokenized + list(chain(*persona))] + history + [[eos]]
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = history

    else:
        #enc_sequence = [[persona_st] + list(chain(*persona))] + [list(chain(*history)) + [eos]]
        enc_sequence = [[persona_st] + landmark_name_tokenized + list(chain(*persona))] + [list(chain(*history)) + [eos]]
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = [list(chain(*history))]


    instance = {}
    instance["input_ids"] = list(chain(*enc_sequence))
    #instance["input_eos"] = len(list(chain(*enc_sequence)))-1
    instance["input_eos"] = 0
    instance["history_list"] = history_list
    instance["dialog"] = list(chain(*dialog))
    instance["decoder_input_ids"] = dec_sequence[:-1]
    instance["lm_labels"] = dec_sequence[1:]
    instance["persona_candidates"] = [[persona_st] + can + [eos] for can in persona_cans]
    #instance["persona_can_idx"] = [len(can)-1 for can in instance["persona_candidates"]]
    instance["persona_can_idx"] = [0 for _ in instance["persona_candidates"]]
    instance["persona_grounding"] = persona_grounding
    #instance["knowledge_candidates"] = [[knowledge_st] + can[:100] + [eos] if len(can) > 100 else [knowledge_st] + can + [eos] for can in knowledge_cans]
    instance["knowledge_candidates"] = [[knowledge_st] + landmark_name_tokenized + can[:100] + [eos] if len(can) > 100 else [knowledge_st] + landmark_name_tokenized + can + [eos] for can in knowledge_cans]
    instance["knowledge_can_idx"] = [len(can)-1 for can in instance["knowledge_candidates"]]
    instance["knowledge_grounding"] = knowledge_answer_idx
    instance["dialog_ID"] = ID
    instance["reply"] = reply
    instance['tot_knowledge'] = paragraphs
    #instance['tot_knowledge_eos'] = [len(p)-1 for p in paragraphs]
    instance['tot_knowledge_eos'] = [0 for _ in paragraphs]
    assert len(instance["decoder_input_ids"]) == len(instance["lm_labels"])

    return instance

def build_input_for_led(args, persona, knowledge, landmark_link, history, persona_cans, persona_grounding, knowledge_cans, knowledge_answer_idx, ID, tokenizer, retrieval_type, sentence_knowledge_vector_dict=None, BM25Okapi = None,  table=None, inference=False):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    eos = tokenizer.eos_token_id
    dec_bos = 2 #tokenizer.decoder_start_token_id
    machine_st = tokenizer.convert_tokens_to_ids(tokenizer.machine_token)
    human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
    persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
    knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)
    bos = 0
    #machine: 50265 human: 50266 persona: 50267 knowledge: 50268 padding: 1 eos: 2

    history, reply = history[:-1], history[-1]
    history = [[human_st if i % 2 == 0 else machine_st] + s for i, s in enumerate(history)]

    knowledge = [para for para in knowledge if len(para) > 0]


    if inference == False:
        if len(knowledge) > 1:
            if retrieval_type == "DPR":
                if args.DPR_train is False:
                    chosen_knowledge = choose_knowledge_dpr(landmark_link, history[-1], tokenizer=tokenizer, table=table)
                elif args.DPR_train is True:
                    chosen_knowledge = choose_knowledge_trained_dpr(landmark_link, history[-1], sentence_knowledge_vector_dict, model_key='bart', tokenizer=tokenizer, table=table)
            elif retrieval_type == "TFIDF":
                chosen_knowledge = choose_knowledge_with_tfidf(knowledge, history[-1], table)
            elif retrieval_type == "TFIDF_sen":
                chosen_knowledge = choose_knowledge_sentence_with_tfidf(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', table=table)
            elif retrieval_type == "BM25":
                chosen_knowledge = choose_knowledge_with_bm25(knowledge, history[-1], BM25Okapi)
            elif retrieval_type == "BM25_sen":
                chosen_knowledge = choose_knowledge_sentence_with_bm25(knowledge, history[-1], sentence_knowledge_vector_dict, landmark_link, model_key='bart', BM25Okapi=BM25Okapi)

        else:
            chosen_knowledge = knowledge[0:5]
    else:
        chosen_knowledge = knowledge_cans[knowledge_answer_idx]

    paragraphs = []
    for para in chosen_knowledge:
        if len(para) > 100:
            short_para = para[:100]
        else:
            short_para = para
        paragraphs.append(short_para)

    landmark_name_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(landmark_link.split('/')[-1]))

    # if testset == False:
    if len(history) == 1:
        # enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history
        enc_sequence = [[bos]] + [landmark_name_tokenized] + [[persona_st] + list(chain(*persona))] + history
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = [[bos]] + history
    else:
        # enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
        enc_sequence = [[bos]] + [landmark_name_tokenized] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
        dec_sequence = [dec_bos] + reply + [eos]
        dialog = [[bos]] + [list(chain(*history))]


    instance = {}
    instance["input_ids"] = list(chain(*enc_sequence))
    #instance["input_eos"] = len(list(chain(*enc_sequence)))-1
    instance["input_eos"] = 0
    instance["dialog"] = list(chain(*dialog))
    instance["decoder_input_ids"] = dec_sequence[:-1]
    instance["lm_labels"] = dec_sequence[1:]
    instance["persona_candidates"] = [[bos] + [persona_st] + can + [eos] for can in persona_cans]
    #instance["persona_can_idx"] = [len(can)-1 for can in instance["persona_candidates"]]
    instance["persona_can_idx"] = [0 for _ in instance["persona_candidates"]]
    instance["persona_grounding"] = persona_grounding
    # instance["knowledge_candidates"] = [[bos] + [knowledge_st] + can[:100] + [eos] if len(can) > 100 else [bos] + [knowledge_st] + can + [eos] for can in knowledge_cans]
    instance["knowledge_candidates"] = [[bos] + [knowledge_st] + landmark_name_tokenized + can[:100] + [eos] if len(can) > 100 else [bos] + [knowledge_st] + landmark_name_tokenized + can + [eos] for can in knowledge_cans]
    instance["knowledge_can_idx"] = [len(can)-1 for can in instance["knowledge_candidates"]]
    instance["knowledge_grounding"] = knowledge_answer_idx
    instance["dialog_ID"] = ID
    instance["reply"] = reply
    instance['tot_knowledge'] = paragraphs
    #instance['tot_knowledge_eos'] = [len(p)-1 for p in paragraphs]
    instance['tot_knowledge_eos'] = [0 for _ in paragraphs]

    assert len(instance["decoder_input_ids"]) == len(instance["lm_labels"])

    return instance



def get_data_loaders(args, tokenizer, get_aug_data=False):
    """ Prepare the dataset for training and evaluation """

    focus = get_dataset_only_train_dev(tokenizer, args.train_dataset_path, args.train_dataset_cache, args.dev_dataset_path, args.dev_dataset_cache, get_aug_data=get_aug_data)

    model_name = args.model_name
    retrieval_type = args.retrieval_type
    if retrieval_type == "DPR":
        if args.DPR_train is False:
            print("table = DPR()")
            if args.use_knowledge_embedidngs is True:
                from retrieval.dpr_simple import DPR_sim
                table = DPR_sim()  ##############33
            else:
                from retrieval.dpr import DPR
                table = DPR()
        elif args.DPR_train is True:
            from retrieval.dpr_for_kr import DPR_for_KR
            print("model = DPR_for_KR()")
            table = DPR_for_KR()
            checkpoint = torch.load(args.DPR_model_path)
            table.load_state_dict(checkpoint)
            table = table.to(args.device)

    elif retrieval_type == "TFIDF" or retrieval_type == "TFIDF_sen":
        from retrieval.tfidf import TfIdf
        table = TfIdf()
    elif retrieval_type == "BM25" or retrieval_type == "BM25_sen":
        from rank_bm25 import BM25Okapi

    if retrieval_type == "TFIDF_sen" or retrieval_type == "BM25_sen" or retrieval_type == "DPR":
        with open(args.landmark_dic, "r", encoding="utf-8") as f:
            sentence_knowledge_vector_dict = json.loads(f.read())


    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in focus.items():
        print(dataset_name, len(dataset))
        aug_question_total = 0
        for dialog in dataset:
            ID = dialog["dialogID"]
            persona = dialog['persona']
            knowledge = dialog['knowledge']
            utterance = dialog['utterance']
            landmark_link = dialog["landmark_link"]
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2*args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_grouding = utt['persona_grounding']
                knowledge_cans = utt['knowledge_candidates']
                knowledge_answer_idx = utt['knowledge_answer_index']
                aug = ['aug_question' in utt.keys()]
                if i == 0 or aug[0] == False:
                    aug = 0
                    aug_data = 1
                else:
                    aug_question = utt['aug_question']
                    aug_question_total += 1
                    aug_data = 2
                for number in range(aug_data):
                    if number == 1:
                        history[-2] = aug_question[0]
                    if model_name == 'BART' or model_name == 'transformer-encdec':
                        if retrieval_type == "BM25_sen":
                            instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                                             persona_cans, persona_grouding,
                                                                             knowledge_cans, knowledge_answer_idx, ID,
                                                                             tokenizer, retrieval_type,
                                                                             sentence_knowledge_vector_dict = sentence_knowledge_vector_dict,
                                                                             BM25Okapi= BM25Okapi,
                                                                             inference=args.inference)
                        elif retrieval_type == "TFIDF_sen":
                            instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                                             persona_cans, persona_grouding,
                                                                             knowledge_cans, knowledge_answer_idx, ID,
                                                                             tokenizer, retrieval_type,
                                                                             sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                                             table=table,  # sent_tokenize=sent_tokenize,
                                                                             inference=args.inference)
                        elif retrieval_type == "BM25":
                            instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                                             persona_cans, persona_grouding,
                                                                             knowledge_cans, knowledge_answer_idx, ID,
                                                                             tokenizer, retrieval_type,
                                                                             BM25Okapi= BM25Okapi,
                                                                             inference=args.inference)

                        elif retrieval_type == "TFIDF" or "DPR":
                            instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                                            persona_cans, persona_grouding,
                                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                                            tokenizer, retrieval_type,
                                                                            table=table,
                                                                            inference=args.inference)

                        else: #DPR-KL train
                             instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                                             persona_cans, persona_grouding,
                                                                             knowledge_cans, knowledge_answer_idx, ID,
                                                                             tokenizer, retrieval_type,
                                                                             sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                                             table=table,
                                                                             inference=args.inference)


                    elif model_name == 'T5':
                        if retrieval_type == "BM25_sen":
                            instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            BM25Okapi=BM25Okapi,
                                                            inference=args.inference)
                        elif retrieval_type == "TFIDF_sen":
                            instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            table=table,  # sent_tokenize=sent_tokenize,
                                                            inference=args.inference)
                        elif retrieval_type == "BM25":
                            instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            BM25Okapi=BM25Okapi,
                                                            inference=args.inference)

                        elif retrieval_type == "TFIDF" or "DPR":
                            instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            table=table,
                                                            inference=args.inference)

                        else:  # DPR-KL train
                            instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            table=table,
                                                            inference=args.inference)


                    elif model_name == 'LED':
                        if retrieval_type == "BM25_sen":
                            instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            BM25Okapi=BM25Okapi,
                                                            inference=args.inference)

                        elif retrieval_type == "TFIDF_sen":
                            instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            table=table,  # sent_tokenize=sent_tokenize,
                                                            inference=args.inference)

                        elif retrieval_type == "BM25":
                            instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            BM25Okapi=BM25Okapi,
                                                            inference=args.inference)

                        elif retrieval_type == "TFIDF" or "DPR":
                            instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            table=table,
                                                            inference=args.inference)

                        else:  # DPR-KL train
                            instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                            persona_cans, persona_grouding,
                                                            knowledge_cans, knowledge_answer_idx, ID,
                                                            tokenizer, retrieval_type,
                                                            sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                            table=table,
                                                            inference=args.inference)

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
        print("num aug: ", aug_question_total)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in CTXT_MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])

    return train_dataset, valid_dataset




def get_testdata_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """

    focus = get_dataset_only_test(tokenizer, args.test_dataset_path, args.test_dataset_cache)

    model_name = args.model_name
    retrieval_type = args.retrieval_type
    if retrieval_type == "DPR":
        if args.DPR_train is False:
            from retrieval.dpr import DPR
            print("table = DPR()")
            table = DPR()  ##############33
        elif args.DPR_train is True:
            from retrieval.dpr_for_kr import DPR_for_KR
            print("model = DPR_for_KR()")
            table = DPR_for_KR()
            checkpoint = torch.load(args.DPR_model_path)
            table.load_state_dict(checkpoint)
            table = table.to(args.device)

    elif retrieval_type == "TFIDF" or retrieval_type == "TFIDF_sen":
        from retrieval.tfidf import TfIdf
        table = TfIdf()
    elif retrieval_type == "BM25" or retrieval_type == "BM25_sen":
        from rank_bm25 import BM25Okapi

    if retrieval_type == "TFIDF_sen" or retrieval_type == "BM25_sen" or retrieval_type == "DPR":
        with open(args.landmark_dic, "r", encoding="utf-8") as f:
            sentence_knowledge_vector_dict = json.loads(f.read())

    logger.info("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    for dataset_name, dataset in focus.items():
        print(dataset_name, len(dataset), "dialogues")
        for dialog in dataset:
            ID = dialog["dialogID"]
            persona = dialog['persona']
            knowledge = dialog['knowledge']
            utterance = dialog['utterance']
            landmark_link = dialog["landmark_link"]
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2*args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_grouding = utt['persona_grounding']
                knowledge_cans = utt['knowledge_candidates']
                knowledge_answer_idx = utt['knowledge_answer_index']

                if model_name == 'BART' or model_name == 'transformer-encdec':
                    if retrieval_type == "BM25_sen":
                        instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                        persona_cans, persona_grouding,
                                                        knowledge_cans, knowledge_answer_idx, ID,
                                                        tokenizer, retrieval_type,
                                                        sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                        BM25Okapi=BM25Okapi,
                                                        inference=args.inference)
                    elif retrieval_type == "TFIDF_sen":
                        instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                        persona_cans, persona_grouding,
                                                        knowledge_cans, knowledge_answer_idx, ID,
                                                        tokenizer, retrieval_type,
                                                        sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                        table=table,  # sent_tokenize=sent_tokenize,
                                                        inference=args.inference)
                    elif retrieval_type == "BM25":
                        instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                        persona_cans, persona_grouding,
                                                        knowledge_cans, knowledge_answer_idx, ID,
                                                        tokenizer, retrieval_type,
                                                        BM25Okapi=BM25Okapi,
                                                        inference=args.inference)

                    elif retrieval_type == "TFIDF" or "DPR":
                        instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                        persona_cans, persona_grouding,
                                                        knowledge_cans, knowledge_answer_idx, ID,
                                                        tokenizer, retrieval_type,
                                                        table=table,
                                                        inference=args.inference)

                    else:  # DPR-KL train
                        instance = build_input_for_bart(args, persona, knowledge, landmark_link, history,
                                                        persona_cans, persona_grouding,
                                                        knowledge_cans, knowledge_answer_idx, ID,
                                                        tokenizer, retrieval_type,
                                                        sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                        table=table,
                                                        inference=args.inference)


                elif model_name == 'T5':
                    if retrieval_type == "BM25_sen":
                        instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                      persona_cans, persona_grouding,
                                                      knowledge_cans, knowledge_answer_idx, ID,
                                                      tokenizer, retrieval_type,
                                                      sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                      BM25Okapi=BM25Okapi,
                                                      inference=args.inference)
                    elif retrieval_type == "TFIDF_sen":
                        instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                      persona_cans, persona_grouding,
                                                      knowledge_cans, knowledge_answer_idx, ID,
                                                      tokenizer, retrieval_type,
                                                      sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                      table=table,  # sent_tokenize=sent_tokenize,
                                                      inference=args.inference)
                    elif retrieval_type == "BM25":
                        instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                      persona_cans, persona_grouding,
                                                      knowledge_cans, knowledge_answer_idx, ID,
                                                      tokenizer, retrieval_type,
                                                      BM25Okapi=BM25Okapi,
                                                      inference=args.inference)

                    elif retrieval_type == "TFIDF" or "DPR":
                        instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                      persona_cans, persona_grouding,
                                                      knowledge_cans, knowledge_answer_idx, ID,
                                                      tokenizer, retrieval_type,
                                                      table=table,
                                                      inference=args.inference)

                    else:  # DPR-KL train
                        instance = build_input_for_t5(args, persona, knowledge, landmark_link, history,
                                                      persona_cans, persona_grouding,
                                                      knowledge_cans, knowledge_answer_idx, ID,
                                                      tokenizer, retrieval_type,
                                                      sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                      table=table,
                                                      inference=args.inference)


                elif model_name == 'LED':
                    if retrieval_type == "BM25_sen":
                        instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                       persona_cans, persona_grouding,
                                                       knowledge_cans, knowledge_answer_idx, ID,
                                                       tokenizer, retrieval_type,
                                                       sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                       BM25Okapi=BM25Okapi,
                                                       inference=args.inference)
                    elif retrieval_type == "TFIDF_sen":
                        instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                       persona_cans, persona_grouding,
                                                       knowledge_cans, knowledge_answer_idx, ID,
                                                       tokenizer, retrieval_type,
                                                       sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                       table=table,  # sent_tokenize=sent_tokenize,
                                                       inference=args.inference)
                    elif retrieval_type == "BM25":
                        instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                       persona_cans, persona_grouding,
                                                       knowledge_cans, knowledge_answer_idx, ID,
                                                       tokenizer, retrieval_type,
                                                       BM25Okapi=BM25Okapi,
                                                       inference=args.inference)

                    elif retrieval_type == "TFIDF" or "DPR":
                        instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                       persona_cans, persona_grouding,
                                                       knowledge_cans, knowledge_answer_idx, ID,
                                                       tokenizer, retrieval_type,
                                                       table=table,
                                                       inference=args.inference)

                    else:  # DPR-KL train
                        instance = build_input_for_led(args, persona, knowledge, landmark_link, history,
                                                       persona_cans, persona_grouding,
                                                       knowledge_cans, knowledge_answer_idx, ID,
                                                       tokenizer, retrieval_type,
                                                       sentence_knowledge_vector_dict=sentence_knowledge_vector_dict,
                                                       table=table,
                                                       inference=args.inference)

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}

    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.pad_token_id)
        for input_name in CTXT_MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])

    return test_dataset



def build_input_persona_test(bertinput, barttokenizer, model_pred_knowledge, all_persona, pred_utterance, gold_utterance):
    bos, eos = barttokenizer.bos_token_id, barttokenizer.eos_token_id
    knowledge_st, persona_st, human_st = barttokenizer.convert_tokens_to_ids(barttokenizer.knowledge_token), barttokenizer.convert_tokens_to_ids(barttokenizer.persona_token), barttokenizer.convert_tokens_to_ids(barttokenizer.human_token)
    input_ids = [bos] + [knowledge_st] + model_pred_knowledge + [persona_st] + list(chain(*all_persona)) + [eos] + pred_utterance + [eos]

    instance = dict()
    instance['input_ids'] = input_ids
    instance['before_refine'] = pred_utterance
    instance['gold_utterance'] = gold_utterance
    instance['model_pred_knowledge'] = model_pred_knowledge
    instance['persona'] = list(chain(*all_persona))


    instance['input_ids_bert'] = bertinput

    return instance

def get_data_for_refine_loaders(args, barttokenizer, berttokenizer, file):
    """ Prepare the dataset for training and evaluation """
    focus = get_dataset_only_test(barttokenizer, args.test_dataset_path, args.test_dataset_cache)
    for dataset_name, dataset in focus.items():
        print(dataset_name, len(dataset), "dialogues")
        persona_list = []
        prev_utterance_list = []
        for dial_idx, dialog in enumerate(dataset):
            persona = dialog['persona']
            utterance = dialog['utterance']
            for utt_i, utt in enumerate(utterance):
                prev_utterance = utt['dialog'][-2]
                prev_utterance_list.append(prev_utterance)
                persona_list.append(persona)


    with open(file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

        dataset_enc = {"test": defaultdict(list)}

        assert len(dataset["text_result"]) == len(prev_utterance_list) == len(persona_list)
        for index, dialogue in enumerate(dataset["text_result"]):
            model_pred_knowledge = barttokenizer.convert_tokens_to_ids(barttokenizer.tokenize(dialogue["model_pred_knowledge"]))
            pred_utterance = barttokenizer.convert_tokens_to_ids(barttokenizer.tokenize(dialogue["pred"]))
            gold_utterance = barttokenizer.convert_tokens_to_ids(barttokenizer.tokenize(dialogue["gold"]))
            persona = persona_list[index]
            prev_utterance = prev_utterance_list[index]
            cls, sep = berttokenizer.cls_token_id, berttokenizer.sep_token_id
            knowledge_st, persona_st = berttokenizer.convert_tokens_to_ids(berttokenizer.knowledge_token), berttokenizer.convert_tokens_to_ids(berttokenizer.persona_token)
            model_pred_knowledge_bert = berttokenizer.convert_tokens_to_ids(berttokenizer.tokenize(dialogue["model_pred_knowledge"]))
            pred_utterance_bert = berttokenizer.convert_tokens_to_ids(berttokenizer.tokenize(dialogue["pred"]))
            persona_bert = [barttokenizer.decode(sent) for sent in persona]
            persona_bert = [berttokenizer.convert_tokens_to_ids(berttokenizer.tokenize(sent)) for sent in persona_bert]

            input_ids_bert = [cls] + [knowledge_st] + model_pred_knowledge_bert + [persona_st] + list(chain(*persona_bert)) + [sep] + pred_utterance_bert + [sep]


            instance = build_input_persona_test(input_ids_bert, barttokenizer, model_pred_knowledge, persona, pred_utterance, gold_utterance)

            for input_name, input_array in instance.items():
                dataset_enc["test"][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}

    for dataset_name, dataset in dataset_enc.items():
        max_l = max(len(x) for x in dataset["input_ids"])
        max_gold_l = max(len(x) for x in dataset["gold_utterance"])
        max_know_l = max(len(x) for x in dataset["model_pred_knowledge"])
        #max_prev_l = max(len(x) for x in dataset["prev_utterance"])
        max_befrefine_l = max(len(x) for x in dataset["before_refine"])
        max_persona_l = max(len(x) for x in dataset["persona"])
        max_l_bert = max(len(x) for x in dataset["input_ids_bert"])

        dataset['input_ids'] = [x + [barttokenizer.pad_token_id] * (max_l - len(x)) for x in dataset['input_ids']]
        dataset['gold_utterance'] = [x + [barttokenizer.pad_token_id] * (max_gold_l - len(x)) for x in dataset['gold_utterance']]
        dataset['model_pred_knowledge'] = [x + [barttokenizer.pad_token_id] * (max_know_l - len(x)) for x in dataset['model_pred_knowledge']]
        #dataset['prev_utterance'] = [x + [berttokenizer.pad_token_id] * (max_prev_l - len(x)) for x in dataset['prev_utterance']]
        dataset['before_refine'] = [x + [barttokenizer.pad_token_id] * (max_befrefine_l - len(x)) for x in dataset['before_refine']]
        dataset['persona'] = [x + [barttokenizer.pad_token_id] * (max_persona_l - len(x)) for x in dataset['persona']]
        dataset['input_ids_bert'] = [x + [berttokenizer.pad_token_id] * (max_l_bert - len(x)) for x in dataset['input_ids_bert']]
        for input_name in ["input_ids", "gold_utterance", "model_pred_knowledge", "before_refine", "persona", "input_ids_bert"]:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            print(input_name, tensor.size())
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])

    return test_dataset