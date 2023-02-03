import json
import jsonlines



# file_list = ["train", "test_random_split", "test_topic_split", "valid_random_split", "valid_topic_split"]
file_list = ["valid"]
for file in file_list:
    ner_data = []
    with open("/home/data/ssh5131/focus_modeling/our_data/"+"test"+"_ours_add_ner_refine.json") as f:
        # for line in f.iter():
        #         ner_data.append(line)
        ner_data = json.load(f)
    # print(len(ner_data))
    # print(ner_data[0])



    # print(ner_data[0]["utterance"][0].keys())
    ner_data = ner_data["data"]

    with open("/home/data/ssh5131/focus_modeling/eval_output/KL0_BM25_BART_beam5_"+"test"+".json") as f:
        pred_datas = json.load(f)
    print(pred_datas.keys())
    pred_datas = pred_datas["text_result"]
    print(len(pred_datas))
    print(pred_datas[0])

    # with open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/"+file+".json") as f:
    #     origin = json.load(f)
    # print(len(origin))
    # print(origin[0])
    i=0

    for dialog in ner_data:

        for utt in dialog["utterance"]:
            pred_data = pred_datas[i]
            utt["input"] = pred_data["input"]
            # utt["input_ids"] = pred_data["input_ids"]
            utt["output"]= pred_data["pred"]
            utt["labels"] = pred_data["gold"]
            i+= 1
        # print(utt)
        # print()

    print(ner_data[0]["utterance"][0].keys())
    print(ner_data[0])
    print(len(pred_datas))
    print(i)

    # print(ner_data[0])
    # breakpoint()
    with open('/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/'+"test"+'.json','w') as f:
        json.dump(ner_data, f)