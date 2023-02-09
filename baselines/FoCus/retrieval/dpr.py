import sys
import os
from transformers import DPRReader, DPRReaderTokenizer
import numpy as np
import torch
import json


class DPR:
    def __init__(self):
        with open("/home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json", "r", encoding="utf-8") as f:
            self.total_dic = json.loads(f.read())

        self.dpr_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')

        self.model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.top_K = 15

    def similarities(self, landmark_link, question, tokenizer):
        """Returns a list of all the [docname, similarity_score] pairs relative to a list of words. """
        # https://github.com/huggingface/transformers/blob/924484ee4a6ebc79426d27eef31a1ee7d13cbb9a/src/transformers/models/dpr/modeling_dpr.py#L228
        self.tokenizer = tokenizer
        # print(self.tokenizer)
        #dec_q = self.tokenizer.decode(question)
        model_input = []
        print(landmark_link)
        enc_knowledge = self.total_dic[landmark_link]
        enc_knowledge_dpr = enc_knowledge["dpr"]
        if self.tokenizer.name_or_path == "facebook/bart-base":
            self.corpus_dict = enc_knowledge["bart"]
        elif self.tokenizer.name_or_path == "t5-base":
            self.corpus_dict = enc_knowledge["t5"]
        dec_q = self.tokenizer.decode(question)
        dpr_enc_q = self.dpr_tokenizer(dec_q)["input_ids"]  ## [CLS] q [SEP]

        for i in enc_knowledge_dpr:
            temp_input = torch.zeros(128)  # model input의 모든 pair의 크기가 같아야해서
            q_k = torch.cat([torch.tensor(dpr_enc_q), torch.tensor(i[1:])])
            len_q_k = min(len(q_k), 128)
            temp_input[:len_q_k] = q_k[:len_q_k]

            model_input.append(temp_input)

        model_input = torch.stack(model_input)
        model_input = model_input.to(self.device).long()


        if model_input.size()[0] > 300:
            model_input = model_input[:300]

        outputs = self.model(model_input)
        # print("sequence_len:  ",outputs["start_logits"].size())

        relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

        sort_rl = np.argpartition(relevance_logits, -self.top_K)

        sort_rl = sort_rl[::-1].tolist()
        print("sort_rl[:self.top_K]:   ", sort_rl[:self.top_K])

        sorted_knowledge = [self.corpus_dict[x] for x in sort_rl[:self.top_K]]
        self.corpus_dict = {}

        return sorted_knowledge
