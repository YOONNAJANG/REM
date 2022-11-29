import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import BartForConditionalGeneration
from torch.nn import Softmax

logger = logging.getLogger(__name__)


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim , 1)  # hiddensize, numclasses
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
            cls_labels=None
            ):


        output_dict = dict()
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias #batch, decseqlen, dim
        cls_logits_cls = outputs['encoder_last_hidden_state'] #batch, encseqlen, dim
        cls_logits =self.summary(cls_logits_cls).squeeze(-1)
        output_dict['cls_logits'] = cls_logits

        masked_lm_loss = None
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            output_dict['lm_loss'] = masked_lm_loss

        cls_loss = None
        if cls_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(cls_logits, cls_labels.type_as(cls_logits))
            softmax = Softmax()
            cls_softmax = softmax(cls_logits)
            _, cls_pred = torch.topk(cls_softmax, k=1, dim=-1)
            cls_acc = torch.sum(cls_labels.type_as(cls_pred)==cls_pred) / (len(cls_labels.view(-1)) * 1.0)

            #output_dict['lm_logits'] = lm_logits
            output_dict['cls_loss'] = cls_loss
            output_dict['cls_acc'] = cls_acc

        return output_dict



