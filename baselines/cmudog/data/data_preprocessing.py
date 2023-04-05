import os
import json

wiki_dict = {}
wiki_input_path = "./WikiData"
files = [file for file in os.listdir(wiki_input_path)]
for i in files:
    with open(os.path.join(wiki_input_path, i), "r") as f:
        data = json.load(f)
    wiki_dict[data["wikiDocumentIdx"]] = data


def concat_user(json_data, int_value):
    data_dict = {"wikiDocumentIdx": json_data["wikiDocumentIdx"]}
    history_list = []
    current_docIdx = 0
    current_speaker = json_data["history"][0]["uid"]
    text = ""
    for each_history in json_data["history"]:
        if current_docIdx != each_history["docIdx"]:
            history_list.append({"docIdx": current_docIdx, "text": text.strip(), "uid": current_speaker})
            current_docIdx = each_history["docIdx"]
            current_speaker = each_history["uid"]
            text = ""

        if current_speaker != each_history["uid"]:
            history_list.append({"docIdx": current_docIdx, "text": text.strip(), "uid": current_speaker})
            current_speaker = each_history["uid"]
            text = each_history["text"]
        else:
            text += " " + each_history["text"]

    history_list.append({"docIdx": current_docIdx, "text": text.strip(), "uid": current_speaker})

    data_dict["history"] = history_list
    # with open(f"./Conversations/concat_test/{int_value}.json", 'w', encoding='utf-8') as make_file:
    #     json.dump(data_dict, make_file, indent="\t")
    return data_dict
    




def data_process(input_path, fname, int_value):
    return_dict = {}
    return_dict["dialogID"] = int_value
    utt_list = []
    with open(os.path.join(input_path, fname), "r") as f:
        json_data = json.load(f)
    data = concat_user(json_data, int_value)
    num_text = len(data["history"])
    wiki = wiki_dict[data["wikiDocumentIdx"]]

    # <2_step>
    # if num_text % 2 == 0:
    #     last_utt_list = [i for i in range(1, num_text + 1, 2)]
    # else:
    #     last_utt_list = [i for i in range(1, num_text, 2)]

    # <1_step>
    last_utt_list = list(range(1, num_text))

    dia_id = 1
    for utt_num in last_utt_list:
        tmp_utt_dict = {}
        text_list =[]
        for u in range(utt_num + 1):
            text_list.append(data["history"][u]["text"])
        tmp_utt_dict[f"dialogue{dia_id}"] = text_list
        if data["history"][utt_num]["docIdx"] in [1, 2, 3]:
            tmp_utt_dict["selected_knowledge"] = wiki[str(data["history"][utt_num]["docIdx"])]
        else:
            knowledge_dict = wiki[str(data["history"][utt_num]["docIdx"])]
            string_list = ""
            for k in knowledge_dict.keys():
                if type(knowledge_dict[k]) is list:
                    string_list += ", " + k + ": [" + ", ".join(knowledge_dict[k]) + "]"
                    # for sent in knowledge_dict[k]:
                    #     string_list = string_list + " " + sent
                elif type(knowledge_dict[k]) is str:
                    string_list += ", " + k + ": " + knowledge_dict[k]
                    # string_list = string_list + " " + knowledge_dict[k]
                else:
                    print(f"type(knowledge_dict[k]) is {type(knowledge_dict[k])}")
                    exit()
            tmp_utt_dict["selected_knowledge"] = string_list.strip()[2:]

        utt_list.append(tmp_utt_dict)
        dia_id += 1
    return_dict["utterance"] = utt_list
    #     print(utt_list)
    return return_dict


if __name__ == '__main__':

    target_file = "test"
    input_path = f"./Conversations/{target_file}"
    output_path = "./Conversations/processed/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = [file for file in os.listdir(input_path)]

    print("There are {} files to process.\nStart processing data ...".format(len(files)))
    total = []
    total_dict = {}
    int_value = 0
    for file in files:
        if file == ".ipynb_checkpoints":
            continue
        print("Preprocessing {} ...".format(file))
        return_dict = data_process(input_path, file, int_value)
        total.append(return_dict)
        int_value += 1
        print("=" * 60)
    total_dict["data"] = total
    #     print(total_dict)
    print("data preprocess done!")

    with open(output_path+f"{target_file}.json", 'w', encoding='utf-8') as make_file:
        json.dump(total_dict, make_file, indent="\t")