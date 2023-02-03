import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import BartForConditionalGeneration
from torch.nn import Sigmoid, Softmax
from datasets import load_metric
logger = logging.getLogger(__name__)


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim , 3)  # hiddensize, numclasses

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
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}

        nSamples = [4813018, 399695, 900778]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.normedWeights = torch.FloatTensor(normedWeights).to("cuda")

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            lm_labels=None,
            ner_labels=None
            ):


        output_dict = dict()
        # print("input_ids.size(): ",input_ids.size()) # [2, 312] -> [bos] [knoweldge token] gk [persona token] ps [human token] history(last)
        # print("cls_labels.size(): ",ner_labels.size()) # [2, 312]
        # print(input_ids)
        if input_ids != None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids)

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim
        ner_logits_cls = outputs['encoder_last_hidden_state']  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits
        output_dict['lm_logits'] = lm_logits


        # hst_index = (input_ids == 50266).nonzero(as_tuple=True)[1]
        # # print("******** hst_idex: ", hst_index)
        # # outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias #batch, decseqlen, dim
        # # ner_logits_cls = outputs['encoder_last_hidden_state'] #batch, encseqlen, dim
        # #
        # # ner_logits_cls = ner_logits_cls[:,:hst_index]
        # # ner_labels = ner_labels[:,:hst_index]
        # # ner_logits =self.summary(ner_logits_cls).squeeze(-1)
        # # output_dict['ner_logits'] = ner_logits

        masked_lm_loss = None
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            output_dict['lm_loss'] = masked_lm_loss

        ner_loss = None
        if ner_labels is not None:

            # loss_fct = CrossEntropyLoss(weight =self.normedWeights, ignore_index=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # cls_loss = loss_fct(cls_logits, cls_labels.type_as(cls_logits))

            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)

            ner_acc = 0
            # print(predictions)
            # print(ner_labels)

            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            # print(true_predictions)
            # print(true_labels)
            # print("----------------")

            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }


            # output_dict['lm_logits'] = lm_logits
            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results
            # breakpoint()
        return output_dict


class BartEncDec_NER_txt(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.max_position_embeddings
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.pad_token_id = config.pad_token_id
        self.max_len = config.max_position_embeddings
        nSamples = [4813018, 399695, 900778]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.normedWeights = torch.FloatTensor(normedWeights).to("cuda")

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            lm_labels=None,
            ner_labels=None
            ):

        output_dict = dict()
        # print("input_ids.size(): ",input_ids.size()) # [2, 312] -> [bos] [knoweldge token] gk [persona token] ps [human token] history(last)
        # print("cls_labels.size(): ",ner_labels.size()) # [2, 312]
        # print(input_ids)
        if input_ids != None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids)

        ner_logits_cls = outputs['encoder_last_hidden_state']  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits


        softmax = Softmax(dim=-1)
        softmax_result, top_ner_result = torch.topk(softmax(ner_logits), 1) #"B":1, "I":2, "O":0,

        result_true = (top_ner_result == 1) + (top_ner_result == 2)

        dec_input_pad_mask = torch.ne(decoder_input_ids, self.pad_token_id)
        dec_input_wo_pad = [r[m] for r, m in zip(decoder_input_ids, dec_input_pad_mask)]
        lm_labels_wo_pad = [r[m] for r, m in zip(lm_labels, dec_input_pad_mask)]

        new_dec_inputs = []
        new_lm_labels = []
        for batch_idx, batch in enumerate(result_true):
            chosen_tok_list = []
            for item_idx, item in enumerate(batch):
                if item == True:
                    chosen_tok_list.append(input_ids[batch_idx][item_idx])
            chosen_tok_list = [torch.tensor(2).to(input_ids.device)] + chosen_tok_list
            chosen_tok_tensor = torch.stack(chosen_tok_list, 0)

            new_dec_input = torch.cat([chosen_tok_tensor, dec_input_wo_pad[batch_idx]])
            new_lm_label = torch.cat([torch.tensor([-100]).repeat(chosen_tok_tensor.size()).to(input_ids.device), lm_labels_wo_pad[batch_idx]])

            pad_len = self.max_len - new_dec_input.size()[0]
            new_dec_input = torch.cat([new_dec_input, torch.tensor([1]).repeat(pad_len).to(input_ids.device)])
            new_lm_label = torch.cat([new_lm_label, torch.tensor([-100]).repeat(pad_len).to(input_ids.device)])

            new_dec_inputs.append(new_dec_input)
            new_lm_labels.append(new_lm_label)

        new_dec_inputs = torch.stack(new_dec_inputs, 0)
        new_lm_labels = torch.stack(new_lm_labels, 0)

        if input_ids != None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=new_dec_inputs)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=new_dec_inputs)

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim
        output_dict['lm_logits'] = lm_logits

        masked_lm_loss = None
        if new_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), new_lm_labels.view(-1))
            output_dict['lm_loss'] = masked_lm_loss




        # hst_index = (input_ids == 50266).nonzero(as_tuple=True)[1]
        # # print("******** hst_idex: ", hst_index)
        # # outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias #batch, decseqlen, dim
        # # ner_logits_cls = outputs['encoder_last_hidden_state'] #batch, encseqlen, dim
        # #
        # # ner_logits_cls = ner_logits_cls[:,:hst_index]
        # # ner_labels = ner_labels[:,:hst_index]
        # # ner_logits =self.summary(ner_logits_cls).squeeze(-1)
        # # output_dict['ner_logits'] = ner_logits

        # masked_lm_loss = None
        # if lm_labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
        #     output_dict['lm_loss'] = masked_lm_loss

        ner_loss = None
        if ner_labels is not None:

            # loss_fct = CrossEntropyLoss(weight =self.normedWeights, ignore_index=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # cls_loss = loss_fct(cls_logits, cls_labels.type_as(cls_logits))

            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)

            ner_acc = 0
            # print(predictions)
            # print(ner_labels)

            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            # print(true_predictions)
            # print(true_labels)
            # print("----------------")

            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }


            # output_dict['lm_logits'] = lm_logits
            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results
            # breakpoint()
        return output_dict
