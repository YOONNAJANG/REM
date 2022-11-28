#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss
from transformers import T5Model, T5ForConditionalGeneration
from transformers import BartModel, BartForConditionalGeneration
from transformers import LEDModel, LEDForConditionalGeneration
from torch.nn import Sigmoid, Softmax

logger = logging.getLogger(__name__)

def generate(input_ids, model, num_beams, num_return_sequences, top_k):
    if top_k == 50:
        do_sample = False
    else:
        do_sample = True
    # noinspection PyPackageRequirements
    output = model.generate(input_ids=input_ids, do_sample=do_sample, num_beams=num_beams, top_k=top_k, no_repeat_ngram_size=2, num_return_sequences=num_return_sequences)
    return output


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class PConcatSummary(nn.Module): # PG때쓰는 ConcatSummary--> KG까지 concat된걸 받아서 하는
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 8, 1)  # hiddensize, numclasses
    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits

class KConcatSummary(nn.Module): # 원래의 ConcatSummary --> KG때 쓰는 ConcatSummary
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 7, 1)  # hiddensize, numclasses
    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits

class ConcatSummary_for_KL_Ori(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 7, 2)  # hiddensize, numclasses
    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits

class ConcatSummary_for_KL_Kg(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 8, 2)  # hiddensize, numclasses
    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits

class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim , 1)  # hiddensize, numclasses
    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits



class Bart_pkgen(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.K_concat_summary = KConcatSummary(emb_dim=config.d_model)
        self.P_concat_summary = PConcatSummary(emb_dim=config.d_model)
        self.O_concat_summary_kl = ConcatSummary_for_KL_Ori(emb_dim=config.d_model)
        self.K_concat_summary_kl = ConcatSummary_for_KL_Kg(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.max_position = config.max_position_embeddings
        self.init_weights()


    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None,
            ):

        #machine = 50265
        #human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        bos_tensor = torch.tensor([bos]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)

        outputs = dict()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            lm_hidden_states = self.model(input_ids=input_ids)['last_hidden_state']
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)

            tot_knowledge_hidden_states = self.model(input_ids=tot_knowledge.view(batch*num_chosen_paragraph, -1))['last_hidden_state'].view(batch, num_chosen_paragraph, -1, embdim)
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(tot_knowledge_hidden_states_batch[j], -2, tot_knowledge_eos_batch[j])
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(input_ids)


            #knowledge candidates
            knowledge_for_persona_list = []
            num_knowledge_can = 10
            softmax = Softmax(dim=-1)
            if knowledge_input_ids is not None:
                knowledge_emb = self.model(input_ids=knowledge_input_ids.view(batch*num_knowledge_can, -1))['last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

                if knowledge_can_idx is not None:
                    knowledge_list = []

                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            knowledge_selected = torch.index_select(knowledge_emb_batch[i], 0, knowledge_can_idx_batch[i]) #####
                            final_rep_knowledge = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), knowledge_selected.type_as(lm_eos_rep)], dim=0)
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)

                        ##### gt_knowledge
                        if knowledge_grounding is not None:
                            gtknowledge_selected = torch.index_select(knowledge_emb_batch, 0, knowledge_grounding[batch_i])
                            gtknowledge_selected = gtknowledge_selected[:,0,:]
                            # print("***************    gtknowledge_selected.size(): ",gtknowledge_selected.size())
                            knowledge_for_persona_list.append(gtknowledge_selected)

                    knowledge_rep = torch.stack(knowledge_list).view(batch*num_knowledge_can, -1)
                    knowledge_logits = self.K_concat_summary(knowledge_rep).view(batch, -1)
                    outputs['knowledge_logits']=knowledge_logits

                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]

                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)

                        knowledge_pred = knowledge_pred[1:-1]
                        all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos
                        if knowledge_grounding is None:
                            for  pred_i in knowledge_pred_idx:
                                predknowledge_for_persona = torch.index_select(knowledge_emb[batch_i], 0, pred_i)
                                predknowledge_for_persona = predknowledge_for_persona[:,0,:] #select the last token embedding
                                # print("***************    predknowledge_for_persona.size(): ", predknowledge_for_persona.size())
                                knowledge_for_persona_list.append(predknowledge_for_persona)

            #persona candidates
            num_persona_can = 5
            if persona_input_ids is not None:
                persona_emb = self.model(input_ids=persona_input_ids.view(batch*num_persona_can,-1))['last_hidden_state'].view(batch, num_persona_can, -1, embdim)
                if persona_can_idx is not None:
                    persona_list_k = [] ## origin CR + KG
                    persona_list_o = [] ##origin CR

                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        persona_emb_batch = persona_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list_k = [] ## origin CR + KG
                        persona_batch_list_o = [] ##origin CR


                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(persona_emb_batch[i], 0, persona_can_idx_batch[i])
                            knowledge_for_persona = knowledge_for_persona_list[batch_i]

                            final_rep_persona = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), persona_selected.type_as(lm_eos_rep)], dim=0) ########여기서 gt_knowledge 가 concat되어야함
                            final_rep_persona_k = torch.cat([final_rep_persona, knowledge_for_persona.type_as(lm_eos_rep)], dim=0)
                            persona_batch_list_k.append(final_rep_persona_k)
                            persona_batch_list_o.append(final_rep_persona)

                        persona_batch_list_k = torch.stack(persona_batch_list_k)
                        persona_batch_list_o = torch.stack(persona_batch_list_o)

                        persona_list_k.append(persona_batch_list_k)
                        persona_list_o.append(persona_batch_list_o)

                    persona_rep_k = torch.stack(persona_list_k).view(batch*num_persona_can, -1)
                    persona_rep_o = torch.stack(persona_list_o).view(batch*num_persona_can, -1)

                    persona_logits = self.K_concat_summary(persona_rep_o).view(batch, -1) ##  도움받는애(원래의 persona_logits)를 persona logits으로

                    persona_logits_k =self.K_concat_summary_kl(persona_rep_k) ## persona + knowledge
                    persona_logits_o = self.O_concat_summary_kl(persona_rep_o) ## persona

                    outputs['persona_logits'] = persona_logits.view(batch, -1) ##  persona_logits은 KG까지 합쳐진걸로

                    persona_pred_softmax = softmax(persona_logits_o.view(batch, num_persona_can, -1))
                    _, k_index_per = torch.topk(persona_pred_softmax,k=1,dim=-1)
                    _ = _.squeeze(-1)
                    k_index_per = k_index_per.squeeze(-1)

                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(k_index_per, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[1:-1])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)



            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i])>0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]


                if new_input_size < int(self.max_position) :
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding]*padding_size).cuda(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)
        dynamic_lm_hidden_states = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)['last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs['dynamic_lm_logits'] = dynamic_lm_logits

        if persona_grounding is not None:
            #loss_fct = BCEWithLogitsLoss()
            loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(6))

            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))

            KLDivloss_fct = KLDivLoss(reduction='batchmean')

            persona_prob_k = torch.nn.functional.softmax(persona_logits_k, dim=1)
            persona_prob_o = torch.nn.functional.log_softmax(persona_logits_o, dim=1)

            kl_loss = KLDivloss_fct(persona_prob_o, persona_prob_k)

            outputs['persona_loss'] = persona_loss
            outputs['kldiv_loss'] = kl_loss

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs['knowledge_loss'] = knowledge_loss
            # print(knowledge_loss)

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['lm_loss'] = lm_loss

        return outputs  # (lm_loss-training), (knowledge_loss), (kldiv_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)




class T5_pkgen(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        #self.model = T5Model(config)
        #self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.K_concat_summary = KConcatSummary(emb_dim=config.d_model)
        self.P_concat_summary = PConcatSummary(emb_dim=config.d_model)
        self.O_concat_summary_kl = ConcatSummary_for_KL_Ori(emb_dim=config.d_model)
        self.K_concat_summary_kl = ConcatSummary_for_KL_Kg(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.max_position = config.n_positions
        self.config = config
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None
    ):

        #machine = 32100
        #human = 32101
        persona = 32102
        knowledge = 32103
        padding = 0
        eos = 1
        num_chosen_paragraph = 5
        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)

        outputs = dict()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            dec_input_lm_eos = shift_tokens_right(input_ids, self.config.pad_token_id,self.config.decoder_start_token_id)
            lm_encoder_hidden_states = self.encoder(input_ids=input_ids)[0]
            lm_decoder_hidden_states = self.decoder(input_ids=dec_input_lm_eos, encoder_hidden_states=lm_encoder_hidden_states)['last_hidden_state']
            batch, seq_len, embdim = lm_decoder_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_decoder_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)

            dec_input_tot_knowledge = shift_tokens_right(tot_knowledge.view(batch*num_chosen_paragraph, -1), self.config.pad_token_id,self.config.decoder_start_token_id)
            tot_knowledge_enc_hidden_states = self.encoder(input_ids=tot_knowledge.view(batch*num_chosen_paragraph, -1))[0]
            tot_knowledge_dec_hidden_states = self.decoder(input_ids=dec_input_tot_knowledge, encoder_hidden_states=tot_knowledge_enc_hidden_states)['last_hidden_state'].view(batch, num_chosen_paragraph, -1, embdim)
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_dec_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(tot_knowledge_hidden_states_batch[j], -2, tot_knowledge_eos_batch[j])
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(input_ids)


            #knowledge candidates
            knowledge_for_persona_list = []
            num_knowledge_can = 10
            softmax = Softmax(dim=-1)
            if knowledge_input_ids is not None:
                dec_input_knowledge_emb = shift_tokens_right(knowledge_input_ids.view(batch*num_knowledge_can, -1), self.config.pad_token_id,self.config.decoder_start_token_id)
                knowledge_enc_emb = self.encoder(input_ids=knowledge_input_ids.view(batch*num_knowledge_can, -1))[0]
                knowledge_dec_emb = self.decoder(input_ids=dec_input_knowledge_emb, encoder_hidden_states=knowledge_enc_emb)['last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

                if knowledge_can_idx is not None:
                    knowledge_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_dec_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            # print("knowledge_can_idx_batch[i].size():   ",knowledge_can_idx_batch[i])
                            knowledge_selected = torch.index_select(knowledge_emb_batch[i], 0, knowledge_can_idx_batch[i]) #####
                            final_rep_knowledge = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), knowledge_selected.type_as(lm_eos_rep)], dim=0)
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)

                        ##### gt_knowledge
                        if knowledge_grounding is not None:
                            gtknowledge_selected = torch.index_select(knowledge_emb_batch, 0, knowledge_grounding[batch_i])
                            gtknowledge_selected = gtknowledge_selected[:,0,:]
                            # print("***************    gtknowledge_selected.size(): ",gtknowledge_selected.size())
                            knowledge_for_persona_list.append(gtknowledge_selected)

                    knowledge_rep = torch.stack(knowledge_list).view(batch*num_knowledge_can, -1)
                    knowledge_logits = self.K_concat_summary(knowledge_rep).view(batch, -1)
                    outputs['knowledge_logits']=knowledge_logits

                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)

                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[:-1]
                        all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos
                        if knowledge_grounding is None:
                            for  pred_i in knowledge_pred_idx:
                                predknowledge_for_persona = torch.index_select(knowledge_dec_emb[batch_i], 0, pred_i)
                                predknowledge_for_persona = predknowledge_for_persona[:,0,:]
                                knowledge_for_persona_list.append(predknowledge_for_persona)

            num_persona_can = 5
            if persona_input_ids is not None:
                dec_input_persona_emb = shift_tokens_right(persona_input_ids.view(batch*num_persona_can,-1), self.config.pad_token_id,self.config.decoder_start_token_id)
                persona_enc_emb = self.encoder(input_ids=persona_input_ids.view(batch*num_persona_can,-1))[0]
                persona_dec_emb = self.decoder(input_ids=dec_input_persona_emb, encoder_hidden_states=persona_enc_emb)['last_hidden_state'].view(batch, num_persona_can, -1, embdim)
                if persona_can_idx is not None:
                    persona_list_k = [] ## origin CR + KG
                    persona_list_o = [] ##origin CR
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        persona_emb_batch = persona_dec_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list_k = [] ## origin CR + KG
                        persona_batch_list_o = [] ##origin CR

                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(persona_emb_batch[i], 0, persona_can_idx_batch[i])
                            knowledge_for_persona = knowledge_for_persona_list[batch_i]

                            final_rep_persona = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), persona_selected.type_as(lm_eos_rep)], dim=0)
                            final_rep_persona_k = torch.cat([final_rep_persona, knowledge_for_persona.type_as(lm_eos_rep)], dim=0)
                            persona_batch_list_k.append(final_rep_persona_k)
                            persona_batch_list_o.append(final_rep_persona)

                        persona_batch_list_k = torch.stack(persona_batch_list_k)
                        persona_batch_list_o = torch.stack(persona_batch_list_o)

                        persona_list_k.append(persona_batch_list_k)
                        persona_list_o.append(persona_batch_list_o)

                    persona_rep_k = torch.stack(persona_list_k).view(batch*num_persona_can, -1)
                    persona_rep_o = torch.stack(persona_list_o).view(batch*num_persona_can, -1)

                    persona_logits = self.K_concat_summary(persona_rep_o).view(batch, -1) ##  persona_logits은 원래 CR --> Ori

                    persona_logits_k =self.K_concat_summary_kl(persona_rep_k)
                    persona_logits_o = self.O_concat_summary_kl(persona_rep_o)

                    outputs['persona_logits'] = persona_logits ##  persona_logits은 KG까지 합쳐진걸로

                    persona_pred_softmax = softmax(persona_logits_o.view(batch, num_persona_can, -1))
                    _, k_index_per = torch.topk(persona_pred_softmax, k=1, dim=-1)
                    _ = _.squeeze(-1)
                    k_index_per = k_index_per.squeeze(-1)

                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(k_index_per, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[:-1])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i])>0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat([new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]


                if new_input_size < int(self.max_position) :
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding]*padding_size).cuda(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)
        dynamic_lm_enc_hidden_states = self.encoder(input_ids=input_ids)[0]
        dynamic_lm_hidden_states = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=dynamic_lm_enc_hidden_states)['last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs['dynamic_lm_logits'] = dynamic_lm_logits

        if persona_grounding is not None:
            #loss_fct = BCEWithLogitsLoss()
            loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(6))
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))

            KLDivloss_fct = KLDivLoss(reduction='batchmean')
            persona_prob_k = torch.nn.functional.softmax(persona_logits_k, dim=1)
            persona_prob_o = torch.nn.functional.log_softmax(persona_logits_o, dim=1)

            kl_loss = KLDivloss_fct(persona_prob_o, persona_prob_k)

            outputs['persona_loss'] = persona_loss
            outputs['kldiv_loss'] = kl_loss

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs['knowledge_loss'] = knowledge_loss
            # print(knowledge_loss)

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['lm_loss'] = lm_loss

        return outputs  # (lm_loss-training), (knowledge_loss), (kldiv_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)




class LED_pkgen(LEDForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]
    def __init__(self, config):
        super().__init__(config)
        #self.plm = LEDForConditionalGeneration(config)
        # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.K_concat_summary = KConcatSummary(emb_dim=config.d_model)
        self.P_concat_summary = PConcatSummary(emb_dim=config.d_model)
        self.O_concat_summary_kl = ConcatSummary_for_KL_Ori(emb_dim=config.d_model)
        self.K_concat_summary_kl = ConcatSummary_for_KL_Kg(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.max_position = config.max_decoder_position_embeddings
        self.config = config
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            input_eos=None,
            only_dial_input_ids=None,
            decoder_input_ids=None,
            persona_input_ids=None,
            knowledge_input_ids=None,
            persona_can_idx=None,
            persona_grounding=None,
            knowledge_can_idx=None,
            knowledge_grounding=None,
            tot_knowledge=None,
            tot_knowledge_eos=None,
            training=None,
            lm_labels=None
    ):

        #machine = 50265
        #human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        bos_tensor = torch.tensor([bos]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)

        outputs = dict()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            lm_hidden_states = self.led(input_ids=input_ids)['last_hidden_state']
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(lm_hidden_states_batch, -2, lm_eos_batch)
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)


            tot_knowledge_hidden_states = self.led(input_ids=tot_knowledge.view(batch*num_chosen_paragraph, -1))['last_hidden_state'].view(batch, num_chosen_paragraph, -1, embdim)
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(tot_knowledge_hidden_states_batch[j], -2, tot_knowledge_eos_batch[j])
                    tot_knowledge_eos_list_batch.append(tot_knowledge_eos_token.squeeze())
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)
            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)
            inctxt_states = torch.cat((lm_eos_rep, knowledge_inctxt_eos_rep), dim=1).type_as(input_ids)


            #knowledge candidates
            knowledge_for_persona_list = []
            num_knowledge_can = 10
            softmax = Softmax(dim=-1)
            if knowledge_input_ids is not None:
                knowledge_emb = self.led(input_ids=knowledge_input_ids.view(batch*num_knowledge_can, -1))['last_hidden_state'].view(batch, num_knowledge_can, -1, embdim)

                if knowledge_can_idx is not None:
                    knowledge_list = []

                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            knowledge_selected = torch.index_select(knowledge_emb_batch[i], 0, knowledge_can_idx_batch[i]) #####
                            final_rep_knowledge = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), knowledge_selected.type_as(lm_eos_rep)], dim=0)
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)

                        ##### gt_knowledge
                        if knowledge_grounding is not None:
                            gtknowledge_selected = torch.index_select(knowledge_emb_batch, 0, knowledge_grounding[batch_i])
                            gtknowledge_selected = gtknowledge_selected[:,0,:]
                            knowledge_for_persona_list.append(gtknowledge_selected)

                    knowledge_rep = torch.stack(knowledge_list).view(batch*num_knowledge_can, -1)
                    knowledge_logits = self.K_concat_summary(knowledge_rep).view(batch, -1)
                    outputs['knowledge_logits']=knowledge_logits

                    knowledge_softmax = softmax(knowledge_logits)
                    _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)

                    all_knowledge_pred = []
                    for batch_i in range(batch):
                        knowledge_pred_idx = k_index_1[batch_i]
                        knowledge_pred = knowledge_input_ids[batch_i][knowledge_pred_idx]
                        mask_knowledge = torch.ne(knowledge_pred, padding)
                        knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                        knowledge_pred = knowledge_pred[1:-1]
                        all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos
                        if knowledge_grounding is None:
                            for  pred_i in knowledge_pred_idx:
                                predknowledge_for_persona = torch.index_select(knowledge_emb[batch_i], 0, pred_i)
                                predknowledge_for_persona = predknowledge_for_persona[:,0,:]
                                knowledge_for_persona_list.append(predknowledge_for_persona)

            num_persona_can = 5
            if persona_input_ids is not None:
                persona_emb = self.led(input_ids=persona_input_ids.view(batch*num_persona_can,-1))['last_hidden_state'].view(batch, num_persona_can, -1, embdim)
                if persona_can_idx is not None:
                    persona_list_k = [] ## origin CR + KG
                    persona_list_o = [] ##origin CR
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        persona_emb_batch = persona_emb[batch_i]
                        persona_can_idx_batch = persona_can_idx[batch_i]
                        persona_batch_list_k = [] ## origin CR + KG
                        persona_batch_list_o = [] ##origin CR

                        for i in range(num_persona_can):
                            persona_selected = torch.index_select(persona_emb_batch[i], 0, persona_can_idx_batch[i])
                            knowledge_for_persona = knowledge_for_persona_list[batch_i]

                            final_rep_persona = torch.cat([inctxt_eos_batch.type_as(lm_eos_rep), persona_selected.type_as(lm_eos_rep)], dim=0) ########여기서 gt_knowledge 가 concat되어야함
                            final_rep_persona_k = torch.cat([final_rep_persona, knowledge_for_persona.type_as(lm_eos_rep)], dim=0)
                            persona_batch_list_k.append(final_rep_persona_k)
                            persona_batch_list_o.append(final_rep_persona)

                        persona_batch_list_k = torch.stack(persona_batch_list_k)
                        persona_batch_list_o = torch.stack(persona_batch_list_o)
                        # print("persona_batch_list_k.size():    ",persona_batch_list_k.size()) # torch.Size([5, 8, 768])
                        persona_list_k.append(persona_batch_list_k)
                        persona_list_o.append(persona_batch_list_o)

                    persona_rep_k = torch.stack(persona_list_k).view(batch*num_persona_can, -1)
                    persona_rep_o = torch.stack(persona_list_o).view(batch*num_persona_can, -1)

                    persona_logits = self.K_concat_summary(persona_rep_o).view(batch, -1) ##  persona_logits은 원래 CR --> Ori

                    persona_logits_k =self.K_concat_summary_kl(persona_rep_k)
                    persona_logits_o = self.O_concat_summary_kl(persona_rep_o)

                    outputs['persona_logits'] = persona_logits ##  persona_logits은 KG까지 합쳐진걸로

                    persona_pred_softmax = softmax(persona_logits_o.view(batch, num_persona_can, -1))
                    _, k_index_per = torch.topk(persona_pred_softmax, k=1, dim=-1)
                    _ = _.squeeze(-1)
                    k_index_per = k_index_per.squeeze(-1)

                    all_persona_pred = []
                    selected_persona_idx = list()
                    for batch_idx, persona_batch in enumerate(torch.eq(k_index_per, 1)):
                        batch_list_idx = list()
                        batch_list = list()
                        for i, can in enumerate(persona_batch):
                            if can == True:
                                batch_list_idx.append(can)
                                persona_selected_now = persona_input_ids[batch_idx][i]
                                mask_persona = torch.ne(persona_selected_now, padding)
                                persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                                batch_list.append(persona_selected_now[1:-1])
                        all_persona_pred.append(batch_list)
                        selected_persona_idx.append(batch_list_idx)

            final_input_list = []
            for batch_i in range(batch):
                only_dial_input_ids_batch = only_dial_input_ids[batch_i]
                mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                if len(all_persona_pred[batch_i])>0:
                    concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                    new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                else:
                    new_persona = None

                new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                if new_persona is not None:
                    new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)
                else:
                    new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)

                new_input_size = new_input.size()[0]


                if new_input_size < int(self.max_position) :
                    padding_size = int(self.max_position) - new_input_size
                    add_padding = torch.tensor([padding]*padding_size).cuda(device)
                    final_input = torch.cat([new_input, add_padding], dim=-1)
                final_input_list.append(final_input)
            input_ids = torch.stack(final_input_list)
        dynamic_lm_hidden_states = self.led(input_ids=input_ids, decoder_input_ids=decoder_input_ids)['last_hidden_state']

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs['dynamic_lm_logits'] = dynamic_lm_logits

        if persona_grounding is not None:
            #loss_fct = BCEWithLogitsLoss()
            loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(6))
            persona_loss = loss_fct(persona_logits.view(batch, -1), persona_grounding.type_as(persona_logits))

            KLDivloss_fct = KLDivLoss(reduction='batchmean')
            persona_prob_k = torch.nn.functional.softmax(persona_logits_k, dim=1)
            persona_prob_o = torch.nn.functional.log_softmax(persona_logits_o, dim=1)

            kl_loss = KLDivloss_fct(persona_prob_o, persona_prob_k)

            outputs['persona_loss'] = persona_loss
            outputs['kldiv_loss'] = kl_loss

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(knowledge_logits.view(batch, -1), knowledge_grounding)
            outputs['knowledge_loss'] = knowledge_loss
            # print(knowledge_loss)

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['lm_loss'] = lm_loss

        return outputs  # (lm_loss-training), (knowledge_loss), (kldiv_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)
