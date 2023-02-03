
import json
from tqdm import tqdm
file_list = "train"
with open(f'/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/{file_list}.json') as fopen:
    data = json.load(fopen)

import re

tag_app = '_Apprentice'
tag_wiz = '_Wizard'
count = 0

total_text = []
total_paras = []
total_sents = []
total_persona =[]
for dialogs in data:
    cur_dialog = dialogs['dialog']
    # index: 0:topic dict; 1: partner dict; 2: self dict
    contexts = []
    for i in range(3):
        contexts.append([])
    texts = []
    facts = []
    sents = []
    paras = []

    no_fact = "no_passages_used"
    for index, dialog in enumerate(cur_dialog):
        # string
        speaker = dialog['speaker']
        is_wizard = speaker.find(tag_wiz)
        is_apprentice = speaker.find(tag_app)
        # string
        text = dialog['text']  ### utterance
        if is_wizard != -1:  ##wizard만 checked_sentence, checked_passage 가지고 있음
            # dict
            checked_sentence = dialog['checked_sentence']
            # dict. the value relates to the key of retrieved_passages
            checked_passage = dialog['checked_passage']
        # list. 7 articles. each article is a dict which the key is title, values is sentence list
        retrieved_passages = dialog['retrieved_passages']
        # list, 7 articles' title
        retrieved_topics = dialog['retrieved_topics']

        # update the retrieved source
        if index == 0:
            contexts[0] = retrieved_passages
        else:
            if is_wizard != -1:
                contexts[2] = retrieved_passages
            if is_apprentice != -1:
                contexts[1] = retrieved_passages

        if index == 0 and is_wizard != -1:
            # the first one and it is wizard, then take knowledge as question
            for key, value in checked_sentence.items():
                texts.append(value)
                facts.append(no_fact)
                sents.append(no_fact)
                paras.append(no_fact)
        texts.append(text)
        if is_wizard != -1:
            # get the chosen sentence
            chosen_sent = no_fact
            # only one sentence here actually
            for key, value in checked_sentence.items():
                chosen_sent = value

            sents.append(chosen_sent)

            # get the chosen passage
            passage_key = ''
            passage_value = ''
            # only one value actually
            for key, value in checked_passage.items():
                passage_key = key
                passage_value = value
            search_passages = contexts[0]
            if passage_key.startswith('partner'):
                search_passages = contexts[1]
            elif passage_key.startswith('self'):
                search_passages = contexts[2]
            chosen_para = no_fact
            for dic_article in search_passages:
                if passage_value in dic_article.keys():
                    chosen_para = ' '.join(dic_article[passage_value])
                    break
            paras.append(chosen_para)

            # get all of the candidates passages
            candidates = []
            for i in range(2, 0, -1):
                context = contexts[i]
                # for context in contexts:
                for dic_article in context:
                    para_fact = []
                    for _, item in dic_article.items():
                        para_fact.append(' '.join(item))
                    if len(para_fact) > 0:
                        para_fact_line = '\t'.join(para_fact)
                        candidates.append(para_fact_line)
            all_facts = '\t'.join(candidates)
            facts.append(all_facts)

        else:
            facts.append(no_fact)
            sents.append(no_fact)
            paras.append(no_fact)

    assert (len(texts) == len(facts))
    assert (len(texts) == len(sents))
    assert (len(texts) == len(paras))
    # print(len(texts))
    # print(texts)
    # breakpoint()

    total_persona.append(dialogs["persona"])
    total_text.append(texts)
    total_paras.append(paras)
    total_sents.append(sents)
print(len(total_text))
print(len(total_persona))

breakpoint()

total = {}
total_data = []
dia_id = 0

for dia_utt, dia_kn, persona in tqdm(zip(total_text, total_sents, total_persona)):
    tmp = {}
    utt_list = []
    tmp["dialogID"] = dia_id
    tmp["persona"] = persona
    tmp["knowledge"] = []
    dia_utt_idx = 0
    for idx in range(1, len(dia_utt), 2):
        tmp_dia_utt = {}

        tmp_dia_utt[f"dialogue{dia_utt_idx + 1}"] = dia_utt[:idx + 1]
        tmp_dia_utt["selected_knowledge"] = dia_kn[idx]
        dia_utt_idx += 1

        utt_list.append(tmp_dia_utt)

    tmp["utterance"] = utt_list
    dia_id += 1

    total_data.append(tmp)
total["data"] = total_data

with open(f"/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/prepro_v2/prepro_{file_list}.json",'w') as f:
    json.dump(total,f)