import json
import jsonlines



# file_list = ["test_topic_split", "valid_random_split", "valid_topic_split", "train", "test_random_split"]
file_list = ["valid_topic_split"]
for file in file_list:
    ner_data = []
    with jsonlines.open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/"+file+"_add_ner_refine.jsonl") as f:
        for line in f.iter():
                ner_data.append(line)
        # ner_data = json.load(f)
    # print(len(ner_data))
    # print(ner_data[0].keys())
    origin = json.load(
        open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/"+file+".json"))



    # print(ner_data[0]["utterance"][0].keys())
    # ner_data = ner_data["data"]

    with open("/home/data/ssh5131/focus_modeling/eval_output/wow/output/new/"+file+"_output_beam1_09k.json") as f:
        pred_datas = json.load(f)
    pred_datas = pred_datas["data"]
    # print(len(pred_datas))
    # print(pred_datas[0])

    # with open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/"+file+".json") as f:
    #     origin = json.load(f)
    # print(len(origin))
    # print(origin[0])
    i=0
    total = []
    ori_i = 0
    for dialog in ner_data[:len(origin)]:
        tmp = {}
        tmp["dialogID"] = dialog["dialogID"]
        tmp["persona"] = dialog["persona"]
        tmp["knowledge"] = dialog["knowledge"]
        ori_i += 1
        if tmp["dialogID"] in [954]:
            # print(pred_datas[i]["labels"])
            # print(pred_datas[i-1]["labels"])eixt

            # print(pred_datas[i+len(dialog["utterance"])]["labels"])
            # print(pred_datas[i+len(dialog["utterance"])-1]["labels"])
            i +=len(dialog["utterance"])
            # breakpoint()
            continue
        dias = []
        print("----------------")

        for utt in dialog["utterance"]:

            # if i> 4335:
            #     continue
            print(i)
            print(ori_i)
            pred_data = pred_datas[i]
            if tmp["persona"] == []:
                tmp["persona"] = [pred_data["persona"]]

            kl = list(utt.keys())
            ori = utt[kl[0]][-1] # 얘는 바뀌어야함
            pred = pred_data["labels"]
            ori_list = ori.split()
            pred_list = pred.split()
            print("*" * 20)
            print(tmp["dialogID"])
            print(ori_list)
            print(pred_list)
            ori_list_ = "".join(ori_list)
            pred_list_ = "".join(pred_list)

            if ori_list_[:10] != pred_list_[:10]:
                print("dddddexi") ## 저장안시키게 해야함

                continue
            else:
                utt["input"] = pred_data["input"]
                utt["input_ids"] = pred_data["input_ids"]
                utt["output"]= pred_data["output"]
                utt["labels"] = pred_data["labels"]
                dias.append(utt)
                i += 1



            # if ori_list[0] != pred_list[0]:
            #     print("dddddexi") ## 저장안시키게 해야함
            #     i -=1

            # print("*" * 20)
            # i +=1









            # print(i)
            # print(pred_data["labels"].strip(), utt[kl[0]][-1].strip(), pred_data["labels"].strip() == utt[kl[0]][-1].strip())
            # print(len(pred_data["labels"].strip()), len(utt[kl[0]][-1].strip()))
            # print(abs(len(pred_data["labels"].strip()) -len(utt[kl[0]][-1].strip())))
            # print()
            #
            # if pred_data["labels"].strip() == utt[kl[0]][-1].strip() or abs(len(pred_data["labels"].strip()) -len(utt[kl[0]][-1].strip())) <3 :
            #     # print(pred_data["labels"], utt[kl[0]][-1])
            #
            #     utt["input"] = pred_data["input"]
            #     # utt["input_ids"] = pred_data["input_ids"]
            #     utt["output"]= pred_data["output"]
            #     utt["labels"] = pred_data["labels"]
            #     dias.append(utt)
            #     i+= 1

        # breakpoint()
        tmp["utterance"] = dias
        # print(tmp)
        total.append(tmp)
        # breakpoint()
        # print(utt)
        # print()

    # print(total[0]["utterance"][0].keys())
    # print(total[0])
    # print(len(pred_datas))
    # print(len(ner_data))
    # print(i)

    print(ner_data[0])
    breakpoint()
    with open('/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_'+file+'.json','w') as f:
        json.dump(total, f)