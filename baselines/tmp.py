import jsonlines
ner_data = []
ner_data_utt = []
count = 0
import json

# ["test_topic_split", "valid_random_split", "valid_topic_split", "train", "test_random_split"]

origin = json.load(open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json"))


infer_result = json.load(open("/home/data/ssh5131/focus_modeling/eval_output/wow/output/new/train_output_beam1_09k.json"))
with jsonlines.open("/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/train_add_ner_refine.jsonl") as f:
    for line in f.iter():
            ner_data.append(line)

print(len(ner_data))
for n in ner_data[:len(origin)]:
    count += len(n["utterance"])
    for k in n["utterance"]:
        ner_data_utt.append(k)
# breakpoint()
print(origin[392])
print(ner_data[392])
breakpoint()
print(infer_result[:100:120])
breakpoint()
print(len(origin))
print(len(ner_data)) #1930
print(len(ner_data_utt)) #4356
print(len(infer_result["data"])) #4356
# print(infer_result["data"][])
breakpoint()
for n, i in zip(ner_data_utt[:2000], infer_result["data"][:2000]):
    kl = list(n.keys())
    print(n[kl[0]][-1])
    print(i["labels"])
    print("*"*20)

    # ['Burritos', 'are', 'so', 'delicious', 'and', 'full', 'of', 'yummy', 'fillings.']