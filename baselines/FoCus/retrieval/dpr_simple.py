import sys
import os
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import numpy as np
import torch
import json


class DPR_sim:
    def __init__(self):
        with open("/home/yoonna/all_landmark_dic.json", "r", encoding="utf-8") as f:
            self.total_dic = json.loads(f.read())

        #with open("/home/mnt/leejeongwoo/project/focus/FoCus_modeling/retrieval/ctx_all_landmark_dic.json", "r", encoding="utf-8") as f:
        #with open("/home/mnt/ssh5131/FoCus_data/toy_dataset/ctx_all_landmark_dic.json", "r", encoding="utf-8") as f:
        with open("/home/mnt/yoonna/focus_modeling/our_data/ctx_all_landmark_dic.json", "r", encoding="utf-8") as f:
            self.embed_dic = json.loads(f.read())

        self.dpr_tokenizer_q = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.top_K = 15

    def similarities(self, landmark_link, question, tokenizer):

       """Returns a list of all the [docname, similarity_score] pairs relative to a list of words. """
       # https://github.com/huggingface/transformers/blob/924484ee4a6ebc79426d27eef31a1ee7d13cbb9a/src/transformers/models/dpr/modeling_dpr.py#L228
       self.tokenizer = tokenizer
       # print(self.tokenizer)
       # dec_q = self.tokenizer.decode(question)
       model_input = []
       print(landmark_link)
       enc_knowledge = self.total_dic[landmark_link]

       if self.tokenizer.name_or_path == "facebook/bart-base":
           self.corpus_dict = enc_knowledge["bart"]
       elif self.tokenizer.name_or_path == "gpt2":
           self.corpus_dict = enc_knowledge["gpt2"]
       elif self.tokenizer.name_or_path == "t5-base":
           self.corpus_dict = enc_knowledge["t5"]
       elif self.tokenizer.name_or_path == "allenai/led-base-16384":
           self.corpus_dict = enc_knowledge["led"]

       dec_q = self.tokenizer.decode(question)
       print("Q:    ", dec_q)
       ## question embedding값 얻기
       dpr_enc_q = self.dpr_tokenizer_q(dec_q, return_tensors="pt")["input_ids"]

       q_embed = self.model(dpr_enc_q.to(self.device)).pooler_output

       ## 저장된 knowledge embed가져오기
       know_embed = self.embed_dic[landmark_link]
       know_embed = torch.tensor(know_embed).to(self.device)
       ## dot
       relevance_logits = torch.matmul(q_embed, torch.transpose(know_embed, 0, 1)).cpu().detach().numpy()
       relevance_logits =  relevance_logits[0]

       sort_rl = np.argpartition(relevance_logits, -self.top_K)

       sort_rl = sort_rl[::-1].tolist()
       print("sort_rl[:self.top_K]:   ", sort_rl[:self.top_K])
       print("top 1: ", self.tokenizer.decode(self.corpus_dict[sort_rl[0]]))

       sorted_knowledge = [self.corpus_dict[x] for x in sort_rl[:self.top_K]]
       self.corpus_dict = {}

       return sorted_knowledge
