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

file_name = "valid"
with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/prev_{file_name}.json", 'r') as read_file:
    ner_data = json.load(read_file)

with open(f"/home/data/leejeongwoo/projects/focus/Refiner/baselines/cmudog/output/epochs_100/epochs_100_{file_name}.json", 'r') as read_file:
    infer_data = json.load(read_file)


c = 0
d = 0
for i in ner_data["data"]:
    new_utt = []
    for u_i, each_turn in enumerate(i["utterance"]):
        if c == len(infer_data["text_result"]) : break

        if each_turn["selected_knowledge"].strip()[:10] == infer_data["text_result"][c]["knowledge"].strip()[:10]:
            each_turn["output"] = infer_data["text_result"][c]["pred"][0]
            new_utt.append(each_turn)
            c += 1
        else:
            d+=1
    i["utterance"] = new_utt

print(ner_data["data"][0])
json.dump(ner_data, open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/{file_name}.json", 'w'))

print(len(ner_data["data"]))
print(len(infer_data["text_result"]))
print(c)
print(d)


