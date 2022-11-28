import torch
import torch.nn as nn
import json
import os
from transformers import DPRReader, DPRReaderTokenizer
from torch.utils.data import Dataset

from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)

with open("all_landmark_dic.json", "r", encoding="utf-8") as f:
    total_dic = json.loads(f.read())

class KR_Dataset(Dataset):
    def __init__(self, total_datas, data_type, max_len, dpr_data_cache=None):
        self.dpr_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.count_over_n = 0

        if dpr_data_cache is not None and os.path.isfile(dpr_data_cache):
            print(f"load {dpr_data_cache}")
            dataset_ckpt = torch.load(dpr_data_cache)
            self.total_model_inputs = dataset_ckpt['data']
        else:
            self.total_model_inputs = []
            pbar = tqdm(total_datas)
            for each_data in pbar:
                landmark_link = each_data["landmark_link"]
                question = each_data["question"]
                golden_knowledge = each_data["golden_knowledge"]

                half_len = int(max_len / 2)

                # make model_input
                model_input_with_gk = []
                model_input_without_gk = []
                enc_knowledge_dpr = total_dic[landmark_link]["dpr"]

                dpr_enc_q = self.dpr_tokenizer(question)["input_ids"][:-1]  ## [CLS] q

                dpr_enc_gk = self.dpr_tokenizer(golden_knowledge)["input_ids"]
                dpr_enc_gk = dpr_enc_gk[1:-1]  ## gk

                for i in enc_knowledge_dpr:
                    if len(model_input_with_gk) >= 400:
                        self.count_over_n += 1
                        break
                    i = i[1:-1]  # [SEP] + knowledge
                    each_knowledge_sentence_vector = [102] + i[1:half_len - len(dpr_enc_q) + 1] + [102]  # [SEP] + knowledge + [SEP]

                    # model_input_without_gk
                    temp_input = torch.zeros(max_len)
                    q_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(each_knowledge_sentence_vector)])
                    len_q_k = min(len(q_k), max_len)
                    temp_input[:len_q_k] = q_k[:len_q_k]
                    model_input_without_gk.append(temp_input)

                    # model_input_with_gk
                    temp_input = torch.zeros(max_len)
                    q_gk_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(dpr_enc_gk[:half_len]),
                                        torch.tensor(each_knowledge_sentence_vector)])
                    len_q_gk_k = min(len(q_gk_k), max_len)
                    temp_input[:len_q_gk_k] = q_gk_k[:len_q_gk_k]
                    model_input_with_gk.append(temp_input)

                pbar.set_postfix({'count_over_n': self.count_over_n})
                model_input_without_gk = torch.stack(model_input_without_gk)
                model_input_with_gk = torch.stack(model_input_with_gk)
                self.total_model_inputs.append({"model_input_without_gk": model_input_without_gk,
                                                "model_input_with_gk": model_input_with_gk,
                                                "landmark_link": landmark_link,
                                                "golden_knowledge": golden_knowledge})

            print("count_over_n:", self.count_over_n)
            dataset_ckpt = {'data': self.total_model_inputs}
            torch.save(dataset_ckpt, f"../data/{data_type}_data_cache.tar.gz")

    def __getitem__(self, i):
        return self.total_model_inputs[i]["model_input_without_gk"], self.total_model_inputs[i]["model_input_with_gk"], self.total_model_inputs[i]["landmark_link"], self.total_model_inputs[i]["golden_knowledge"]

    def __len__(self):
        return len(self.total_model_inputs)


class DPR_for_KR(nn.Module):
    def __init__(self):
        super(DPR_for_KR, self).__init__()

        self.model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
        self.device = torch.device("cuda")

    def similarities(self, model_input):
        # model_input = torch.stack(model_input)
        model_input = model_input.to(self.device).long()

        # print("print(model_input.size())")
        # print(model_input.size())
        outputs = self.model(model_input)
        relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

        return relevance_logits

    def forward(self,
                model_input_without_gk,
                model_input_with_gk=None):
        loss = None
        pure_logits_with_gk = None

        if model_input_with_gk is not None:
            pure_logits_with_gk = self.similarities(model_input_with_gk)
            logits_with_gk = torch.from_numpy(pure_logits_with_gk)

        pure_logits_without_gk = self.similarities(model_input_without_gk)
        logits_without_gk = torch.from_numpy(pure_logits_without_gk)

        if model_input_with_gk is not None:
            unsup_criterion = nn.KLDivLoss(reduction='batchmean')
            with torch.no_grad():
                prob_with_gk = torch.nn.functional.softmax(logits_with_gk, dim=-1)
            prob_without_gk = torch.nn.functional.softmax(logits_without_gk, dim=-1)
            loss = unsup_criterion(prob_without_gk.log(), prob_with_gk)

        return (loss, pure_logits_with_gk, pure_logits_without_gk)
