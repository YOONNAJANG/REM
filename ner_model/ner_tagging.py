import json
from collections import defaultdict

from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)
from setproctitle import setproctitle
import copy
setproctitle("suhyun")
# pip install flair
from flair.data import Sentence
from flair.models import SequenceTagger
# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")


# ###############################################
# ################  chatgpt  ####################
# ###############################################
# total = []
# chatgpt = json.load(open("/home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours.json"))
# for i, data in enumerate(chatgpt["data"]):
#     tmp = {}
#     tmp["dialogID"] = i
#     utt_dict = {}
#     dia_his = data["dialogue_history"]
#
#     dia_his.append(data["our_response"])
#     utt_dict["dialogue1"] = dia_his
#     utt_dict["selected_knowledge"] = data["knowledge"]
#     utt_dict["chatgpt_bad_response"] = data["chatgpt_bad_reaponse"]
#
#
#
#     tmp["utterance"] = [utt_dict]
#
#     for u_i, each_turn in enumerate(tmp["utterance"]):
#         format = {'LOC': {"keyword": [], "persona_index": {"p_1": [], "p_2": [], "p_3": [],
#                                                            "p_4": [],
#                                                            "p_5": []
#                                                            },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   'MISC': {"keyword": [],
#                            "persona_index": {
#                                "p_1": [],
#                                "p_2": [],
#                                "p_3": [],
#                                "p_4": [],
#                                "p_5": []
#                            },
#                            "knowledge_index": [],
#                            "response_index": []},
#                   'PER': {"keyword": [],
#                           "persona_index": {
#                               "p_1": [],
#                               "p_2": [],
#                               "p_3": [],
#                               "p_4": [],
#                               "p_5": []
#                           },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   'ORG': {"keyword": [],
#                           "persona_index": {
#                               "p_1": [],
#                               "p_2": [],
#                               "p_3": [],
#                               "p_4": [],
#                               "p_5": []
#                           },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   }
#         ### response
#         sentence = Sentence(each_turn[f"dialogue{u_i + 1}"][-1])
#         tagger.predict(sentence)
#         for entity in sentence.get_spans('ner'):
#             if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
#                     format[entity.get_label("ner").value]["keyword"]:
#                 # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
#                 format[entity.get_label("ner").value]["keyword"].append(entity.text)
#             format[entity.get_label("ner").value]["response_index"].append((entity.start_position, entity.end_position))
#
#         ### knowledge
#         # knowledge_answer_index = each_turn["knowledge_answer_index"]
#         sentence = Sentence(each_turn["selected_knowledge"])
#         tagger.predict(sentence)
#         for entity in sentence.get_spans('ner'):
#             if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
#                     format[entity.get_label("ner").value]["keyword"]:
#                 # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
#                 format[entity.get_label("ner").value]["keyword"].append(entity.text)
#             format[entity.get_label("ner").value]["knowledge_index"].append(
#                 (entity.start_position, entity.end_position))
#         each_turn["NER_tagging"] = format
#
#
#
#
#
#     total.append(tmp)
#
# with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json", 'w', encoding='utf-8') as make_file:
#     json.dump(total, make_file, indent="\t")



# ###############################################
# ################  wow  ####################
# ###############################################
file_name = "test_random_split" # train, test_topic_split,  test_random_split, valid_topic_split,  valid_random_split
# file_name = "train"
# file_name = "test_topic_split"
# file_name = "valid_topic_split"
# file_name = "valid_random_split"

with open(f"/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/{file_name}.json", 'r') as read_file:
    focus_data = json.load(read_file)
with open(f"/home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/{file_name}_output_beam1_09k.json", 'r') as read_file:
    infer_result = json.load(read_file)


total_data = {'data': []}
total_ner_set = defaultdict(set)
# with open(file, "r", encoding="utf-8") as f:
#     dataset = json.loads(f.read())
#     dataset_enc = dict()
#     dataset_enc[name] = list()

total = []
dataset_enc = dict()
dataset_enc["train"] = list()

for data in focus_data:
    persona = data["persona"]  # sentence
    new_dialogue = {"dialog": []}
    for each_dialog in data["dialog"]:
        utt_enc = dict()
        speaker = each_dialog["speaker"]  # 0_Wizard, 1_Wizard, 0_Apprentice, 1_Apprentice
        text = each_dialog["text"]  # utterance

        if "checked_sentence" in each_dialog.keys():
            if each_dialog["speaker"].split("_")[1] != "Wizard":
                print('each_dialog["speaker"].split("_")[1] == "Wizard"')
                exit()

            if len(each_dialog["checked_sentence"]) > 1:
                print(len(each_dialog["checked_sentence"]))
                exit()

            if ('no_passages_used' in each_dialog["checked_sentence"].keys()) or (
                    len(each_dialog["checked_sentence"]) == 0):
                checked_sentence = 'no_passages_used'
            #                 no_knowledge_count += 1
            elif ('no_passages_used' not in each_dialog["checked_sentence"].keys()) and (
                    len(each_dialog["checked_sentence"]) == 1):
                checked_sentence = list(each_dialog["checked_sentence"].values())[0]
            #                 knowledge_count += 1
            else:
                print('no_passages_used ?, len(each_dialog["checked_sentence"]) ?')
                exit()

        else:
            if each_dialog["speaker"].split("_")[1] != "Apprentice":
                print('each_dialog["speaker"].split("_")[1] != "Apprentice"')
                exit()

            checked_sentence = 'no_passages_used'
        #             app_knowledge_count += 1

        utt_enc["speaker"] = speaker
        utt_enc["text"] = text
        utt_enc["checked_sentence"] = checked_sentence
        #         print(utt_enc)

        new_dialogue["dialog"].append(utt_enc)
    dataset_enc["train"].append(new_dialogue)

total_list = []
dataset_dict = {}
dataset_list = []
check_list = []
infer_num = 0
for dataset_name, dataset in dataset_enc.items():

    for n, data in enumerate(dataset):
        tmp_dict = {}
        checked_sentences = []
        history_sent = []
        history = []

        for each_dialog in data['dialog']:
            history_sent.append(each_dialog['text'])
            history.append([each_dialog['speaker'], each_dialog['text']])
            checked_sentences.append(each_dialog['checked_sentence'])

        if history[0][0] == "0_Wizard":  # wizard starts
            history = history[1:]
            checked_sentences = checked_sentences[1:]

        history_list = []
        history_data = []
        for i, each_utterance in enumerate(history):
            k = [checked_sentences[i]]
            each_utterance.extend(k)

            history_list.append(each_utterance)
            if i % 2 == 1:
                history_now = copy.deepcopy(history_list)
                history_data.append(history_now)

        utt_list = []
        for i, dia in enumerate(history_data):
            utt_dict = {}
            text_list = []
            format = {'LOC': {"keyword": [],
                                                "persona_index": {
                                                    "p_1": [],
                                                    "p_2": [],
                                                    "p_3": [],
                                                    "p_4": [],
                                                    "p_5": []
                                                },
                                                "knowledge_index": [],
                                                "response_index": []},
                                        'MISC': {"keyword": [],
                                                 "persona_index": {
                                                     "p_1": [],
                                                     "p_2": [],
                                                     "p_3": [],
                                                     "p_4": [],
                                                     "p_5": []
                                                 },
                                                 "knowledge_index": [],
                                                 "response_index": []},
                                        'PER': {"keyword": [],
                                                "persona_index": {
                                                    "p_1": [],
                                                    "p_2": [],
                                                    "p_3": [],
                                                    "p_4": [],
                                                    "p_5": []
                                                },
                                                "knowledge_index": [],
                                                "response_index": []},
                                        'ORG': {"keyword": [],
                                                "persona_index": {
                                                    "p_1": [],
                                                    "p_2": [],
                                                    "p_3": [],
                                                    "p_4": [],
                                                    "p_5": []
                                                },
                                                "knowledge_index": [],
                                                "response_index": []},
                                        }
            for text in dia:
                text_list.append(text[1])
            utt_dict[f"dialogue{i}"] = text_list
            utt_dict["selected_knowledge"] = dia[-1][-1]


            sentence = Sentence(text_list[-1])
            tagger.predict(sentence)


            for entity in sentence.get_spans('ner'):
                if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in format[entity.get_label("ner").value]["keyword"]:
                    # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                    format[entity.get_label("ner").value]["keyword"].append(entity.text)
                format[entity.get_label("ner").value]["response_index"].append((entity.start_position, entity.end_position))
#
            sentence = Sentence(utt_dict["selected_knowledge"])
            tagger.predict(sentence)

            for entity in sentence.get_spans('ner'):
                if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                        format[entity.get_label("ner").value]["keyword"]:
                    # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                    format[entity.get_label("ner").value]["keyword"].append(entity.text)
                format[entity.get_label("ner").value]["knowledge_index"].append((entity.start_position, entity.end_position))
            utt_dict["NER_tagging"] = format

            utt_dict["output"] = infer_result["data"][infer_num]["output"]
            utt_list.append(utt_dict)
            infer_num += 1

        total_list.extend(history_data)
        tmp_dict["dialogID"] = n
        tmp_dict["utterance"] = utt_list
        dataset_list.append(tmp_dict)
        check_list.extend(utt_list)
    dataset_dict["data"] = dataset_list
print(len(check_list))
print(len(total_list))
print(dataset_list[0])

with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/{file_name}.json", 'w', encoding='utf-8') as make_file:
    json.dump(dataset_list, make_file, indent="\t")

#
# ###############################################
# ################  cmudog  ####################
# ###############################################
# file_name = "train"
# with open(f"/home/data/ssh5131/focus_modeling/others/cmudog/processed/{file_name}.json", 'r') as read_file:
#     focus_data = json.load(read_file)
#
# total_data = {'data': []}
# total_ner_set = defaultdict(set)
# print(len(focus_data["data"]))
# for each_dialog in tqdm(focus_data['data']):
#     # ner_set = defaultdict(set)
#     # print("-------------------------------------------")
#     # for each_persona in each_dialog['persona']:
#     #     # print(each_persona)
#     #     sentence = Sentence(each_persona)
#     #     tagger.predict(sentence)
#     #     print(sentence)
#     #     for entity in sentence.get_spans('ner'):
#     #         ner_set[entity.get_label("ner").value].add(entity.text)
#     #
#     #         print(ner_set)
#     #
#
#     for u_i, each_turn in enumerate(each_dialog["utterance"]):
#
#         format = {'LOC': {"keyword": [],
#                           "persona_index": {
#                               "p_1": [],
#                               "p_2": [],
#                               "p_3": [],
#                               "p_4": [],
#                               "p_5": []
#                           },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   'MISC': {"keyword": [],
#                            "persona_index": {
#                                "p_1": [],
#                                "p_2": [],
#                                "p_3": [],
#                                "p_4": [],
#                                "p_5": []
#                            },
#                            "knowledge_index": [],
#                            "response_index": []},
#                   'PER': {"keyword": [],
#                           "persona_index": {
#                               "p_1": [],
#                               "p_2": [],
#                               "p_3": [],
#                               "p_4": [],
#                               "p_5": []
#                           },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   'ORG': {"keyword": [],
#                           "persona_index": {
#                               "p_1": [],
#                               "p_2": [],
#                               "p_3": [],
#                               "p_4": [],
#                               "p_5": []
#                           },
#                           "knowledge_index": [],
#                           "response_index": []},
#                   }
#
#         ### response
#         sentence = Sentence(each_turn[f"dialogue{u_i + 1}"][-1])
#         tagger.predict(sentence)
#         for entity in sentence.get_spans('ner'):
#             if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in format[entity.get_label("ner").value]["keyword"]:
#                 # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
#                 format[entity.get_label("ner").value]["keyword"].append(entity.text)
#             format[entity.get_label("ner").value]["response_index"].append((entity.start_position, entity.end_position))
#
#
#         #
#         # ### persona
#         # for p_i, each_persona in enumerate(each_dialog['persona']):
#         #     # sentence = Sentence(each_persona)
#         #     # #     tagger.predict(sentence)
#         #     sentence = Sentence(each_persona)
#         #     tagger.predict(sentence)
#         #     for entity in sentence.get_spans('ner'):
#         #         if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in format[entity.get_label("ner").value]["keyword"]:
#         #             format[entity.get_label("ner").value]["keyword"].append(entity.text)
#         #         format[entity.get_label("ner").value]["persona_index"][f'p_{p_i+1}'].append((entity.start_position, entity.end_position))
#
#         ### knowledge
#         # knowledge_answer_index = each_turn["knowledge_answer_index"]
#         sentence = Sentence(each_turn["selected_knowledge"])
#         tagger.predict(sentence)
#         for entity in sentence.get_spans('ner'):
#             if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
#                     format[entity.get_label("ner").value]["keyword"]:
#                 # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
#                 format[entity.get_label("ner").value]["keyword"].append(entity.text)
#             format[entity.get_label("ner").value]["knowledge_index"].append((entity.start_position, entity.end_position))
#         each_turn["NER_tagging"] = format
#
#     total_data["data"].append(each_dialog)
#     # for k, v in ner_set.items():
#     #     ner_set[k] = list(v)
#     #     total_ner_set[k].update(ner_set[k])
#     #
#     # each_dialog['ner_set'] = ner_set
#     # total_data['data'].append(each_dialog)
#
# print(len(total_data["data"]))
#
# with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/{file_name}.json", 'w', encoding='utf-8') as make_file:
#     json.dump(total_data, make_file, indent="\t")
# # #
# # #
# # # for k, v in total_ner_set.items():
# # #     total_ner_set[k] = list(v)
