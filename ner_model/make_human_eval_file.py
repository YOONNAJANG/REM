from tqdm import tqdm
import json
from collections import defaultdict
from setproctitle import setproctitle
import csv
import random

import copy
setproctitle("yoonna")

# #WOW bart base
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.0_wow_bart-base_refine.json", 'r') as file1:
#     not_refined = json.load(file1)['text_result']
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.5_wow_bart-base_refine.json", 'r') as file2:
#     refined = json.load(file2)['text_result']
#
# assert len(not_refined) == len(refined)
#
# result_list = []
#
# for before, after in zip(not_refined, refined):
#     if after['refine'] == 'True':
#         # input = before['only_input']
#         knowledge = before['knoweldge']
#         if 'no_passages_used' in knowledge:
#             continue
#         before_refine = before['pred'][0]
#         after_refine = after['pred'][0]
#         result_list.append([knowledge, before_refine, after_refine])
#
# print('len: ', len(result_list))
#
# random.shuffle(result_list)
# result_list = result_list[:100]
#
# header = ['Knowledge', 'model A', 'model B', 'Fluent', 'Factually correct', 'Well-paraphrase']
#
# with open('wow_human_eval.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',)
#     writer.writerow(header)
#     for item in result_list:
#         writer.writerow(item)
#

# #CMUDOG bart base
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.0_cmu_bart-base_refine.json", 'r') as file1:
#     not_refined = json.load(file1)['text_result']
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.5_cmu_bart-base_refine.json", 'r') as file2:
#     refined = json.load(file2)['text_result']
#
# assert len(not_refined) == len(refined)
#
# result_list = []
#
# for before, after in zip(not_refined, refined):
#     if after['refine'] == 'True':
#         # input = before['only_input']
#         knowledge = before['knoweldge']
#         before_refine = before['pred'][0]
#         after_refine = after['pred'][0]
#         result_list.append([knowledge, before_refine, after_refine])
#
# print('len: ', len(result_list))
#
# random.shuffle(result_list)
# result_list = result_list[:100]
#
# header = ['Knowledge', 'model A', 'model B', 'Fluent', 'Factually correct', 'Well-paraphrase']
#
# with open('cmu_human_eval.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',)
#     writer.writerow(header)
#     for item in result_list:
#         writer.writerow(item)
#

# #FoCus bart base
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.0_focus_bart-base_refine.json", 'r') as file1:
#     not_refined = json.load(file1)['text_result']
# with open("/home/data/yoonna/Refiner/ner_model/human_eval_data/threshold_0.5_focus_bart-base_refine.json", 'r') as file2:
#     refined = json.load(file2)['text_result']
#
# assert len(not_refined) == len(refined)
#
# result_list = []
#
# for before, after in zip(not_refined, refined):
#     if after['refine'] == 'True':
#         # input = before['only_input']
#         knowledge = before['knoweldge']
#         if 'no_passages_used' in knowledge:
#             continue
#         before_refine = before['pred'][0]
#         after_refine = after['pred'][0]
#         result_list.append([knowledge, before_refine, after_refine])
#
# print('len: ', len(result_list))
#
# random.shuffle(result_list)
# result_list = result_list[:100]
#
# header = ['Knowledge', 'model A', 'model B', 'Fluent', 'Factually correct', 'Well-paraphrase']
#
# with open('focus_human_eval.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',)
#     writer.writerow(header)
#     for item in result_list:
#         writer.writerow(item)
#
