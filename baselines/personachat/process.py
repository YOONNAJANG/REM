import os
import logging
import json
import collections
from pprint import pformat
from argparse import ArgumentParser
from collections import Counter
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
from data_utils import make_logdir, get_gpt2_data_loaders, get_t5_data_loaders, get_bart_data_loaders, add_special_tokens, add_special_tokens_gpt2

logger = logging.getLogger(__file__)



def get_dataset(original_path, revised_path, dnli_path):
    with open(original_path, "r", encoding="utf-8") as original_f:
        original_f = json.loads(original_f.read())
        print("original data: ", original_f.keys(), len(original_f['test']))
    with open(revised_path, "r", encoding="utf-8") as revised_f:
        revised_f = json.loads(revised_f.read())
        print("revised data: ", revised_f.keys(), len(revised_f['test']))
    with open(dnli_path, "r", encoding="utf-8") as dnli_f:
        dnli_f = json.loads(dnli_f.read())
        #print("dnli dat: ", dnli_f.keys(), len(dnli_f['test']))
        print('dnli data:', len(dnli_f))
    return original_f, revised_f, dnli_f


def preprocess():
    from setproctitle import setproctitle
    setproctitle("Yoonna")

    original_dataset_path = "/home/yoonna/persona_chat/data/persona-chat/persona_chat_original.json"
    revised_dataset_path = "/home/yoonna/persona_chat/data/persona-chat/persona_chat_revised.json"
    #dnli_dataset_path = "/home/yoonna/persona_chat/data/DNLI/dnli_triples_i_or_none.json"
    dnli_dataset_path = "/home/yoonna/persona_chat/data/DNLI/dnli_triples.json"

    original_data, revised_data, dnli_data = get_dataset(original_dataset_path, revised_dataset_path, dnli_dataset_path)
    #
    # print('dnli: ', dnli_data.keys())
    # for key, data in dnli_data.items():
    #     print('key: ', key, len(data))
    #     print('data: ', data[0])
    #     for data_item in data:
    #         sent_item = data_item['sent']
    sent_triple_dict = dict()
    for data in dnli_data:
        sent_item = data['sent']
        triple_item = data['triple']
        sent_triple_dict[sent_item] = triple_item
    print('sent triple dict keys : ', sent_triple_dict.keys())

    for key, data in original_data.items():
        #print('key', key)
        #print('data len', len(data))
        for dialogue_item in data:
            #print('item keys: ', dialogue_item.keys())
            personality = dialogue_item['personality']
            personality_triples = []
            for persona in personality:
                if persona in sent_triple_dict.keys():
                    print('persona: ', persona)
            #print('personality: ', personality)
            utterances = dialogue_item['utterances']
            #print('utterances: ', len(utterances))
            #break



if __name__ == "__main__":
    preprocess()
