import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import BartForConditionalGeneration
from torch.nn import Sigmoid, Softmax

logger = logging.getLogger(__name__)


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim , 6)  # hiddensize, numclasses
        # ner_label_map = {"B":1, "I":2,"O":0, tokenizer.persona_token:3,tokenizer.knowledge_token:4, tokenizer.bos_token:5} ### knowledge_st, persona_st, bos

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class BartEncDec(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            lm_labels=None,
            ner_labels=None
            ):


        output_dict = dict()
        # print("input_ids.size(): ",input_ids.size()) # [2, 312] -> [bos] [knoweldge token] gk [persona token] ps [human token] history(last)
        # print("cls_labels.size(): ",ner_labels.size()) # [2, 312]
        # print(input_ids)
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim
        output_dict['lm_logits'] = lm_logits
        ner_logits_cls = outputs['encoder_last_hidden_state']  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits





        hst_index = (input_ids == 50266).nonzero(as_tuple=True)[1]
        # print("******** hst_idex: ", hst_index)
        # outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias #batch, decseqlen, dim
        # ner_logits_cls = outputs['encoder_last_hidden_state'] #batch, encseqlen, dim
        #
        # ner_logits_cls = ner_logits_cls[:,:hst_index]
        # ner_labels = ner_labels[:,:hst_index]
        # ner_logits =self.summary(ner_logits_cls).squeeze(-1)
        # output_dict['ner_logits'] = ner_logits


        masked_lm_loss = None
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            output_dict['lm_loss'] = masked_lm_loss

        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # cls_loss = loss_fct(cls_logits, cls_labels.type_as(cls_logits))


            ner_loss = loss_fct(ner_logits.view(-1, 6), ner_labels.view(-1).long())
            # softmax = Softmax()
            # ner_softmax = softmax(ner_logits)
            # _, ner_pred = torch.topk(ner_softmax, k=1, dim=-1)
            # print("ner_logits.size():  ",ner_logits.size())
            # print("ner_labels.size():  ",ner_labels.size())
            # ner_acc = torch.sum(ner_labels.type_as(ner_pred) == ner_pred) / (len(ner_labels.view(-1)) * 1.0)
            ner_acc = 0
            for i in range(ner_logits.shape[0]):
                # print("hst_index[i]:  ",hst_index[i])
                logits_clean = ner_logits[i][:hst_index[i].item()]
                label_clean = ner_labels[i][:hst_index[i].item()]
                # print(logits_clean.size())
                # print(label_clean.size())

                predictions = logits_clean.argmax(dim=1)
                # print(predictions)
                # print(label_clean)
                acc = (predictions == label_clean).float().mean()
                ner_acc += acc

            # output_dict['lm_logits'] = lm_logits
            output_dict['ner_loss'] = ner_loss
            output_dict['ner_acc'] = ner_acc
            # breakpoint()
        return output_dict

#
# class BartEncDec(BartForConditionalGeneration):
#     _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.summary = Summary(emb_dim=config.d_model)
#         self.max_position = config.max_position_embeddings
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             decoder_input_ids=None,
#             lm_labels=None,
#             cls_labels=None
#             ):
#
#
#         output_dict = dict()
#         outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias #batch, decseqlen, dim
#         cls_logits_cls = outputs['encoder_last_hidden_state'] #batch, encseqlen, dim
#         cls_logits =self.summary(cls_logits_cls).squeeze(-1)
#         output_dict['cls_logits'] = cls_logits
#
#         masked_lm_loss = None
#         if lm_labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
#             output_dict['lm_loss'] = masked_lm_loss
#
#         cls_loss = None
#         if cls_labels is not None:
#             # loss_fct = BCEWithLogitsLoss(ignore_index=-1)
#             loss_fct = BCEWithLogitsLoss()
#             cls_loss = loss_fct(cls_logits, cls_labels.type_as(cls_logits)) #2, 312
#             sigmoid = Sigmoid()
#             cls_pred_sig = sigmoid(cls_logits)
#             cls_pred = (cls_pred_sig > 0.5).float()
#             cls_acc = torch.sum(cls_labels.type_as(cls_pred)==cls_pred) / (len(cls_labels.view(-1)) * 1.0)
#
#             #output_dict['lm_logits'] = lm_logits
#             output_dict['cls_loss'] = cls_loss
#             output_dict['cls_acc'] = cls_acc
#
#         return output_dict
#


