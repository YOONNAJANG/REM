import json

# ["test_topic_split", "valid_random_split", "valid_topic_split", "train", "test_random_split"]

origin = json.load(open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json"))

for i, data in enumerate(origin):
    print(data["dialog"][0]["speaker"])
    if data["dialog"][0]["speaker"][2:] == "Wizard":
        print(i)