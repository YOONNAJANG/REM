import json
import random

trainfile = "/home/yoonna/Downloads/train_focus.json"
train_augfile = "/home/yoonna/Downloads/train_focus_augmented.json"

def main():
    with open(trainfile, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())["data"]
        print('dataset len: ', len(dataset)) #12484
        valid_ours = dataset[:1000]
        train_ours = dataset[1000:]
        valid_ours_dict = dict()
        valid_ours_dict['data'] = valid_ours
        train_ours_dict = dict()
        train_ours_dict['data'] = train_ours
    with open("/home/yoonna/Downloads/train_ours.json", 'w') as our_train:
        json.dump(train_ours_dict, our_train)
    with open("/home/yoonna/Downloads/valid_ours.json", 'w') as our_valid:
        json.dump(valid_ours_dict, our_valid)

    with open(train_augfile, "r", encoding="utf-8") as f:
        dataset_aug = json.loads(f.read())["data"]
        print('dataset len: ', len(dataset_aug)) #12484
        valid_ours_aug = dataset_aug[:1000]
        train_ours_aug = dataset_aug[1000:]
        valid_ours_dict_aug = dict()
        valid_ours_dict_aug['data'] = valid_ours_aug
        train_ours_dict_aug = dict()
        train_ours_dict_aug['data'] = train_ours_aug
    with open("/home/yoonna/Downloads/train_ours_augmented.json", 'w') as our_train_aug:
        json.dump(train_ours_dict_aug, our_train_aug)
    with open("/home/yoonna/Downloads/valid_ours_augmented.json", 'w') as our_valid_aug:
        json.dump(valid_ours_dict_aug, our_valid_aug)

if __name__ == "__main__":
    main()
