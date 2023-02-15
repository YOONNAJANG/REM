import json
from collections import defaultdict

from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)
from setproctitle import setproctitle

setproctitle("suhyun")
# pip install flair
from flair.data import Sentence
from flair.models import SequenceTagger
# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

file_name = "train"
with open(f"/home/data/ssh5131/focus_modeling/others/cmudog/processed/{file_name}.json", 'r') as read_file:
    focus_data = json.load(read_file)

total_data = {'data': []}
total_ner_set = defaultdict(set)
print(len(focus_data["data"]))
for each_dialog in tqdm(focus_data['data']):
    # ner_set = defaultdict(set)
    # print("-------------------------------------------")
    # for each_persona in each_dialog['persona']:
    #     # print(each_persona)
    #     sentence = Sentence(each_persona)
    #     tagger.predict(sentence)
    #     print(sentence)
    #     for entity in sentence.get_spans('ner'):
    #         ner_set[entity.get_label("ner").value].add(entity.text)
    #
    #         print(ner_set)
    #

    for u_i, each_turn in enumerate(each_dialog["utterance"]):

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

        ### response
        sentence = Sentence(each_turn[f"dialogue{u_i + 1}"][-1])
        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):
            if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in format[entity.get_label("ner").value]["keyword"]:
                # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                format[entity.get_label("ner").value]["keyword"].append(entity.text)
            format[entity.get_label("ner").value]["response_index"].append((entity.start_position, entity.end_position))


        #
        # ### persona
        # for p_i, each_persona in enumerate(each_dialog['persona']):
        #     # sentence = Sentence(each_persona)
        #     # #     tagger.predict(sentence)
        #     sentence = Sentence(each_persona)
        #     tagger.predict(sentence)
        #     for entity in sentence.get_spans('ner'):
        #         if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in format[entity.get_label("ner").value]["keyword"]:
        #             format[entity.get_label("ner").value]["keyword"].append(entity.text)
        #         format[entity.get_label("ner").value]["persona_index"][f'p_{p_i+1}'].append((entity.start_position, entity.end_position))

        ### knowledge
        # knowledge_answer_index = each_turn["knowledge_answer_index"]
        sentence = Sentence(each_turn["selected_knowledge"])
        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):
            if len(format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                    format[entity.get_label("ner").value]["keyword"]:
                # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                format[entity.get_label("ner").value]["keyword"].append(entity.text)
            format[entity.get_label("ner").value]["knowledge_index"].append((entity.start_position, entity.end_position))
        each_turn["NER_tagging"] = format

    total_data["data"].append(each_dialog)
    # for k, v in ner_set.items():
    #     ner_set[k] = list(v)
    #     total_ner_set[k].update(ner_set[k])
    #
    # each_dialog['ner_set'] = ner_set
    # total_data['data'].append(each_dialog)

print(len(total_data["data"]))

with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/{file_name}.json", 'w', encoding='utf-8') as make_file:
    json.dump(total_data, make_file, indent="\t")
# #
# #
# # for k, v in total_ner_set.items():
# #     total_ner_set[k] = list(v)
