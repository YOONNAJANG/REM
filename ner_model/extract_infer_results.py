import pandas as pd
import json
import os

path = "/home/data/ssh5131/focus_modeling/eval_output/wow_refiner"
file_list = [file for file in os.listdir(path) if file.endswith('.json')]
print(file_list)

for file_name in file_list:

    infer_result = json.load(open(os.path.join(path,file_name)))

    key = ['chrF++', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'ppl', 'ner_acc','ner_f1', 'factcc_result', 'dae_result', 'dist1_result', 'dist2_result']
    data = []
    for k in key:
        data.append(infer_result[k])

    total = {}
    for k, v in zip(key, data):
        total[k] =[v]

    result_df = pd.DataFrame(total)
    result_df.to_csv(os.path.join(path, file_name+"inference_result.csv"))