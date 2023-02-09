import torch
import torch.nn as nn
import json
import random
import argparse
import numpy as np
from setproctitle import setproctitle

from transformers import DPRReader, DPRReaderTokenizer
from transformers import BartTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm as std_tqdm
from functools import partial

tqdm = partial(std_tqdm, dynamic_ncols=True)

from datasets import load_metric

bert_score_metric = load_metric('bertscore')

# device ###############################
if torch.cuda.is_available():
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    n_gpu = 0
print("# device: {}".format(device))
########################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_model_path",
    type=str,
    default="./models/model_20220308_184748/epoch_1.pt",
    help="load_model_path",
)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


class DPR_for_KR(nn.Module):
    def __init__(self):
        super(DPR_for_KR, self).__init__()

        with open("/mnt/raid6/leejeongwoo/sub_project/focus/FoCus_modeling_server9/python_tf_idf/tvt_landmark_dic.json",
                  "r", encoding="utf-8") as f:
            self.total_dic = json.loads(f.read())

        self.dpr_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)

    def similarities(self, model_input):
        model_input = torch.stack(model_input)
        model_input = model_input.to(device).long()

        outputs = self.model(model_input)
        relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

        return relevance_logits

    def forward(self,
                landmark_link,
                question,
                golden_knowledge):
        max_len = 128
        half_len = int(max_len / 2)

        # make model_input
        model_input_with_gk = []
        model_input_without_gk = []
        enc_knowledge_dpr = self.total_dic[landmark_link]["dpr"]

        dpr_enc_q = self.dpr_tokenizer(question)["input_ids"]  ## [CLS] q [SEP]
        dpr_enc_gk = self.dpr_tokenizer(golden_knowledge)["input_ids"]  ## [CLS] gk [SEP]
        if len(dpr_enc_gk) > 512:
            print("len(dpr_enc_gk):", len(dpr_enc_gk))

        for i in enc_knowledge_dpr:
            if len(model_input_with_gk) >= 300:
                break

            each_knowledge_sentence_vector = i[1:half_len - len(dpr_enc_q) + 1]

            # model_input_with_gk
            temp_input = torch.zeros(max_len)
            q_gk_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(dpr_enc_gk[:half_len]),
                                torch.tensor(each_knowledge_sentence_vector)])
            len_q_gk_k = min(len(q_gk_k), max_len)
            temp_input[:len_q_gk_k] = q_gk_k[:len_q_gk_k]
            model_input_with_gk.append(temp_input)

            # model_input_without_gk
            temp_input = torch.zeros(max_len)
            q_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(each_knowledge_sentence_vector)])
            len_q_k = min(len(q_k), max_len)
            temp_input[:len_q_k] = q_k[:len_q_k]
            model_input_without_gk.append(temp_input)

        pure_logits_with_gk = self.similarities(model_input_with_gk)
        logits_with_gk = torch.from_numpy(pure_logits_with_gk)

        pure_logits_without_gk = self.similarities(model_input_without_gk)
        logits_without_gk = torch.from_numpy(pure_logits_without_gk)

        unsup_criterion = nn.KLDivLoss(reduction='batchmean')
        with torch.no_grad():
            prob_with_gk = torch.nn.functional.softmax(logits_with_gk, dim=-1)
        prob_without_gk = torch.nn.functional.softmax(logits_without_gk, dim=-1)

        loss = unsup_criterion(prob_without_gk.log(), prob_with_gk)

        return (loss, pure_logits_with_gk, pure_logits_without_gk)


class KR_Dataset(Dataset):
    def __init__(self, total_datas):
        self.total_datas = total_datas

    def __getitem__(self, i):
        return self.total_datas[i]["landmark_link"], self.total_datas[i]["question"], self.total_datas[i][
            "golden_knowledge"]

    def __len__(self):
        return len(self.total_datas)


def cacluate_bertscore_with_gt(chosen_knowledge1, groundtruth):
    result = bert_score_metric.compute(predictions=chosen_knowledge1, references=groundtruth, lang='en')
    return result['f1'][0]


if __name__ == "__main__":
    setproctitle("focus_bertscore")

    with open("tvt_landmark_dic.json", "r", encoding="utf-8") as f:
        tvt_landmark_dic = json.loads(f.read())

    print("\n# prepare model")
    model = DPR_for_KR()
    checkpoint = torch.load(args.load_model_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    focus_valid_data = []
    with open("../data/pretty_valid_focus.json", 'r') as read_file:
        json_data = json.load(read_file)
    for each_dialog in tqdm(json_data["data"]):
        for u_i, each_turn in enumerate(each_dialog["utterance"]):
            focus_valid_data.append({"landmark_link": each_dialog["landmark_link"],
                                     "question": "<human> " + each_turn[f"dialogue{u_i + 1}"][-2],
                                     "golden_knowledge": each_turn["knowledge_candidates"][
                                         each_turn["knowledge_answer_index"]]})

    print("\n# prepare valid data")
    valid_dataset = KR_Dataset(focus_valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    print(f"len(valid_dataset): {len(valid_dataset)}")
    print(f"len(valid_dataloader): {len(valid_dataloader)}")

    print("\n# prepare bart_tokenizer")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    model.eval()
    total_bert_score = 0
    count = 0
    pbar = tqdm(valid_dataloader, desc="validation")
    for landmark_link, question, golden_knowledge in pbar:
        with torch.no_grad():
            outputs = model(landmark_link=landmark_link[0],
                            question=question[0],
                            golden_knowledge=golden_knowledge[0])

        enc_knowledge = tvt_landmark_dic[landmark_link[0]]
        bart_dict = enc_knowledge["bart"]
        sort_rl = np.argpartition(outputs[2], -1)
        sort_rl = sort_rl[::-1].tolist()
        sorted_knowledge = [bart_dict[x] for x in sort_rl[:1]]
        bart_dict = {}

        chosen_knowledge1 = sorted_knowledge[0]
        dec_chosen_knowledge1 = bart_tokenizer.decode(chosen_knowledge1[1:-1])  # <s>, </s> 토큰 제거
        bert_score = cacluate_bertscore_with_gt([dec_chosen_knowledge1], [golden_knowledge[0]])
        if bert_score < 0 or bert_score > 1:
            print("bert_score < 0 or bert_score > 1:", bert_score)
        total_bert_score += bert_score
        count += 1
        pbar.set_postfix({'total_bert_score': total_bert_score / count})

    print("total_bert_score:", total_bert_score / len(focus_valid_data))