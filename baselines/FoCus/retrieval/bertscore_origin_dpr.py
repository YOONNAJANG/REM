import argparse
from setproctitle import setproctitle

from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)

from transformers import DPRReader, DPRReaderTokenizer
import numpy as np
import torch
import json
from transformers import BartTokenizer

from datasets import load_metric
bert_score_metric = load_metric('bertscore')

class DPR:
    def __init__(self):
        with open("/mnt/raid6/leejeongwoo/sub_project/focus/FoCus_modeling_server9/python_tf_idf/tvt_landmark_dic.json", "r", encoding="utf-8") as f:
            self.total_dic = json.loads(f.read())

        self.dpr_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')

        self.model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.top_K = 1      # top1

    def similarities(self, landmark_link, question):
        """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""

        # self.tokenizer = tokenizer
        model_input = []
        enc_knowledge = self.total_dic[landmark_link]
        enc_knowledge_dpr = enc_knowledge["dpr"]
        # if self.tokenizer.name_or_path == "facebook/bart-base":
        self.corpus_dict = enc_knowledge["bart"]
        # elif self.tokenizer.name_or_path == "gpt2":
            # self.corpus_dict = enc_knowledge["gpt2"]
        # dec_q = self.tokenizer.decode(question)
        dec_q = question
        dpr_enc_q = self.dpr_tokenizer(dec_q)["input_ids"]  ## [CLS] q [SEP]

        for i in enc_knowledge_dpr:
            temp_input = torch.zeros(128)
            q_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(i[1:])])
            len_q_k = min(len(q_k), 128)
            temp_input[:len_q_k] = q_k[:len_q_k]
            model_input.append(temp_input)

        model_input = torch.stack(model_input)
        model_input = model_input.to(self.device).long()
        # print("model_input.size():   ", model_input.size()) ### (doc_sen_num, seq_len)
        # print(model_input)

        if model_input.size()[0] > 300:
            model_input = model_input[:300]
            # print("model_input.size():   ", model_input.size())
        outputs = self.model(model_input)
        # print("sequence_len:  ",outputs["start_logits"].size())

        relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

        sort_rl = np.argpartition(relevance_logits, -self.top_K)

        sort_rl = sort_rl[::-1].tolist()
        # print("sort_rl[:self.top_K]:   ", sort_rl[:self.top_K])

        sorted_knowledge = [self.corpus_dict[x] for x in sort_rl[:self.top_K]]
        self.corpus_dict = {}

        return sorted_knowledge


def choose_knowledge(landmark_link, question, table):

    results = table.similarities(landmark_link, question)
    # chosen_knowledge = results[:10]
    chosen_knowledge = results[0]

    return chosen_knowledge


def cacluate_bertscore_with_gt(chosen_knowledge1, groundtruth):
    result = bert_score_metric.compute(predictions=chosen_knowledge1, references=groundtruth, lang='en')
    return result['f1'][0]


if __name__ == "__main__":
    setproctitle("focus_bertscore")
    table = DPR()

    focus_valid_data = []
    with open("../data/pretty_valid_focus.json", 'r') as read_file:
        json_data = json.load(read_file)
    for each_dialog in tqdm(json_data["data"]):
        for u_i, each_turn in enumerate(each_dialog["utterance"]):
            focus_valid_data.append({"landmark_link": each_dialog["landmark_link"],
                                     "question": "<human> " + each_turn[f"dialogue{u_i+1}"][-2],
                                     "golden_knowledge": each_turn["knowledge_candidates"][each_turn["knowledge_answer_index"]]})

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    total_bert_score = 0
    count = 0
    pbar = tqdm(focus_valid_data)
    for each_data in pbar:
        chosen_knowledge1 = choose_knowledge(each_data["landmark_link"], each_data["question"], table)
        dec_chosen_knowledge1 = bart_tokenizer.decode(chosen_knowledge1[1:-1])    # remove <s>, </s> token
        bert_score = cacluate_bertscore_with_gt([dec_chosen_knowledge1], [each_data["golden_knowledge"]])
        if bert_score < 0 or bert_score > 1:
            print("bert_score < 0 or bert_score > 1:", bert_score)
        total_bert_score += bert_score
        count += 1
        pbar.set_postfix({'total_bert_score': total_bert_score/count})

    print("total_bert_score:", total_bert_score / len(focus_valid_data))