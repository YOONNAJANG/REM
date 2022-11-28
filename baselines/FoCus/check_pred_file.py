import json
import random

file = "/home/mnt/yoonna/focus_modeling/output/find_lambda_1_1_1.json"


def main():
    with open(file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())["text_result"]
        print('dataset len: ', len(dataset))
        for i, item in enumerate(dataset):
            input = item['input']
            gold = item['gold']
            pred = item['pred']
            #model_pred_knowledge = item['model_pred_knowledge']
            if len(gold) <= 1 or len(pred) <=1:
                print(i, "th data: ")
                print("gold: ", len(gold), gold)
                print("pred: ", len(pred), pred)




if __name__ == "__main__":
    main()
