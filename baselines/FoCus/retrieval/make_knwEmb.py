from transformers import DPRReader, DPRReaderTokenizer, DPRContextEncoderTokenizer, DPRContextEncoder
import json
import torch
from setproctitle import setproctitle
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm as std_tqdm
from functools import partial

tqdm = partial(std_tqdm, dynamic_ncols=True)

# 디바이스 설정 ########################
if torch.cuda.is_available():
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    n_gpu = 0
print("# device: {}".format(device))
########################################


class DPR_emb_Dataset(Dataset):
    def __init__(self, total_sen_k, tokenizer, max_seq_length):
        self.total_sen_k = total_sen_k
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length



    def __getitem__(self, i):
        each_data = self.total_sen_k[i]
        input_ids = dpr_tokenizer(each_data, return_tensors="pt", max_length=self.max_seq_length, padding="max_length",
                                  truncation=True)["input_ids"]
        return input_ids


    def __len__(self):
        return len(self.total_sen_k)

if __name__ == '__main__':
    setproctitle("make_dpr_vector_json")

    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', return_dict=True)
    model = model.to(device)

    data_name = ["/home/mnt/ssh5131/FoCus_data/our_data/train_ours.json", "/home/mnt/ssh5131/FoCus_data/our_data/valid_ours.json", "/home/mnt/ssh5131/FoCus_data/our_data/test_ours.json"]
    dataset_enc = dict()
    for each_file in data_name:
        with open(each_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
            pbar = tqdm(dataset["data"])
            for dialogue in pbar:
                total_sen_k = []
                knowledge = dialogue["knowledge"]
                landmark = dialogue["landmark_link"]
                print(landmark)
                print(len(knowledge))
                for k in knowledge:
                    sen_k = k.split(". ")
                    total_sen_k.extend(sen_k)
                print(len(total_sen_k))
                max_seq_length = 512
                batch_size = 32
                inference_dataset = DPR_emb_Dataset(total_sen_k, dpr_tokenizer, max_seq_length=max_seq_length)
                inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

                total_embedding = []
                for in_i, input_ids in enumerate(inference_dataloader):
                    in_i += 1
                    pbar.set_postfix({'current': f"{in_i}/{len(inference_dataloader)}"})
                    input_ids = input_ids.to(device)
                    # print("input_ids:", type(input_ids), input_ids.size())
                    # print(input_ids)

                    embeddings = model(input_ids.view([-1, max_seq_length])).pooler_output
                    # print("embeddings:", type(embeddings), embeddings.size())
                    # print(embeddings)

                    total_embedding.extend(embeddings.tolist())
                    # print("total_embedding:", type(total_embedding), len(total_embedding))
                    # print("total_embedding[0]:", type(total_embedding[0]), len(total_embedding[0]))
                print("len(total_embedding):   ",len(total_embedding))
                dataset_enc[landmark] = total_embedding
    with open("/home/mnt/ssh5131/FoCus_data/our_data/ctx_all_landmark_dic.json", "w") as json_file:
        json.dump(dataset_enc, json_file)

    # for i, (k, v) in enumerate(dataset_enc.items()):
    #     print('\n', k)
    #     print("total_embedding:", type(v), len(v))
    #     print("total_embedding[0]:", type(v[0]), len(v[0]))
    #     print("total_embedding[0][0]:", type(v[0][0]), v[0][0])

