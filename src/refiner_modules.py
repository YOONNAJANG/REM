import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartPretrainedModel, BartDecoder, BartConfig, BartLearnedPositionalEmbedding, BartEncoderLayer, _expand_mask
from transformers import T5ForConditionalGeneration
import random, math
from typing import Optional

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    Seq2SeqModelOutput,
)
from torch.nn import Softmax
from datasets import load_metric
logger = logging.getLogger(__name__)

class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim , 3)  # hiddensize, numclasses

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

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            labels=None,
            ner_labels=None
            ):

        output_dict = dict()

        if inputs_embeds == None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids)

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim
        ner_logits_cls = outputs['encoder_last_hidden_state']  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            output_dict['loss'] = masked_lm_loss

        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)
            
            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

            output_dict['logits'] = lm_logits
            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results

        return output_dict



class BartEncDec_NER_explicit(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.max_position_embeddings
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.pad_token_id = config.pad_token_id

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            labels=None,
            ner_labels=None
            ):

        output_dict = dict()

        if inputs_embeds == None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids)

        ner_logits_cls = outputs['encoder_last_hidden_state']  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits

        softmax = Softmax(dim=-1)
        _, top_ner_result = torch.topk(softmax(ner_logits), 1) #"B":1, "I":2, "O":0,

        result_true = (top_ner_result == 1) + (top_ner_result == 2)

        dec_input_pad_mask = torch.ne(decoder_input_ids, self.pad_token_id)
        dec_input_wo_pad = [r[m] for r, m in zip(decoder_input_ids, dec_input_pad_mask)]
        lm_labels_wo_pad = [r[m] for r, m in zip(labels, dec_input_pad_mask)]

        new_dec_inputs = []
        new_lm_labels = []
        for batch_idx, batch in enumerate(result_true):
            chosen_tok_list = []
            for item_idx, item in enumerate(batch):
                if item == True:
                    chosen_tok_list.append(input_ids[batch_idx][item_idx])
            chosen_tok_list = [torch.tensor(2).to(input_ids.device)] + chosen_tok_list
            chosen_tok_tensor = torch.stack(chosen_tok_list, 0)
            chosen_tok_tensor_mask = (chosen_tok_tensor!=1).nonzero()
            chosen_tok_tensor = chosen_tok_tensor[chosen_tok_tensor_mask].squeeze(-1)[:512]

            new_dec_input = torch.cat([chosen_tok_tensor, dec_input_wo_pad[batch_idx]])
            new_lm_label = torch.cat([torch.tensor([-100]).repeat(chosen_tok_tensor.size()).to(input_ids.device), lm_labels_wo_pad[batch_idx]])
            pad_len = self.max_position - new_dec_input.size()[0]

            new_dec_input = torch.cat([new_dec_input, torch.tensor([1]).repeat(pad_len).to(input_ids.device)])
            new_lm_label = torch.cat([new_lm_label, torch.tensor([-100]).repeat(pad_len).to(input_ids.device)])

            new_dec_inputs.append(new_dec_input)
            new_lm_labels.append(new_lm_label)

        new_dec_inputs = torch.stack(new_dec_inputs, 0)
        new_lm_labels = torch.stack(new_lm_labels, 0)

        if inputs_embeds == None:
            outputs = self.model(input_ids=input_ids, decoder_input_ids=new_dec_inputs)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, decoder_input_ids=new_dec_inputs)

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim
        output_dict['logits'] = lm_logits

        masked_lm_loss = None
        if new_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), new_lm_labels.view(-1))
            output_dict['loss'] = masked_lm_loss


        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)
            
            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results

        return output_dict


class T5EncDec(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.n_positions
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            labels=None,
            ner_labels=None
            ):

        output_dict = dict()

        if inputs_embeds == None:
            enc_outputs = self.encoder(input_ids=input_ids)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_states)
        else:
            enc_outputs = self.encoder(inputs_embeds=inputs_embeds)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_states)

        lm_logits = self.lm_head(decoder_outputs[0])  # batch, decseqlen, dim
        ner_logits_cls = enc_outputs[0]  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            output_dict['loss'] = masked_lm_loss

        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)
            
            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

            output_dict['logits'] = lm_logits
            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results

        return output_dict


class T5EncDec_NER_explicit(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.n_positions
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.pad_token_id = config.pad_token_id

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            labels=None,
            ner_labels=None
            ):

        output_dict = dict()

        if inputs_embeds == None:
            enc_outputs = self.encoder(input_ids=input_ids)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_states)
        else:
            enc_outputs = self.encoder(inputs_embeds=inputs_embeds)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_states)

        ner_logits_cls = enc_outputs[0]  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)
        output_dict['ner_logits'] = ner_logits

        softmax = Softmax(dim=-1)
        _, top_ner_result = torch.topk(softmax(ner_logits), 1) #"B":1, "I":2, "O":0,

        result_true = (top_ner_result == 1) + (top_ner_result == 2)

        dec_input_pad_mask = torch.ne(decoder_input_ids, self.pad_token_id)
        dec_input_wo_pad = [r[m] for r, m in zip(decoder_input_ids, dec_input_pad_mask)]
        lm_labels_wo_pad = [r[m] for r, m in zip(labels, dec_input_pad_mask)]

        new_dec_inputs = []
        new_lm_labels = []
        for batch_idx, batch in enumerate(result_true):
            chosen_tok_list = []
            for item_idx, item in enumerate(batch):
                if item == True:
                    chosen_tok_list.append(input_ids[batch_idx][item_idx])
            chosen_tok_list = [torch.tensor(2).to(input_ids.device)] + chosen_tok_list
            chosen_tok_tensor = torch.stack(chosen_tok_list, 0)
            chosen_tok_tensor_mask = (chosen_tok_tensor!=1).nonzero()
            chosen_tok_tensor = chosen_tok_tensor[chosen_tok_tensor_mask].squeeze(-1)[:512]

            new_dec_input = torch.cat([chosen_tok_tensor, dec_input_wo_pad[batch_idx]])
            new_lm_label = torch.cat([torch.tensor([-100]).repeat(chosen_tok_tensor.size()).to(input_ids.device), lm_labels_wo_pad[batch_idx]])
            pad_len = self.max_position - new_dec_input.size()[0]

            new_dec_input = torch.cat([new_dec_input, torch.tensor([1]).repeat(pad_len).to(input_ids.device)])
            new_lm_label = torch.cat([new_lm_label, torch.tensor([-100]).repeat(pad_len).to(input_ids.device)])

            new_dec_inputs.append(new_dec_input)
            new_lm_labels.append(new_lm_label)

        new_dec_inputs = torch.stack(new_dec_inputs, 0)
        new_lm_labels = torch.stack(new_lm_labels, 0)

        if inputs_embeds == None:
            enc_outputs = self.encoder(input_ids=input_ids)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=new_dec_inputs, encoder_hidden_states=hidden_states)
        else:
            enc_outputs = self.encoder(inputs_embeds=inputs_embeds)
            hidden_states = enc_outputs[0]
            decoder_outputs = self.decoder(input_ids=new_dec_inputs, encoder_hidden_states=hidden_states)

        lm_logits = self.lm_head(decoder_outputs[0]) # batch, decseqlen, dim
        output_dict['logits'] = lm_logits


        masked_lm_loss = None
        if new_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), new_lm_labels.view(-1))
            output_dict['loss'] = masked_lm_loss


        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)
            
            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

            output_dict['ner_loss'] = ner_loss
            output_dict['ner_results'] = ner_results

        return output_dict



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


class Bart_imp_Encoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.summary = Summary(emb_dim=config.d_model)


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        ner_logits_cls = hidden_states  # batch, encseqlen, dim
        ner_logits = self.summary(ner_logits_cls).squeeze(-1)

        softmax = Softmax(dim=-1)
        _, top_ner_result = torch.topk(softmax(ner_logits), 1) #"B":1, "I":2, "O":0,

        result_true = (top_ner_result == 1) + (top_ner_result == 2)
        new_enc_outputs = []
        for batch_idx, batch in enumerate(result_true):
            chosen_vec_list = []
            for item_idx, item in enumerate(batch):
                if item == True:
                    chosen_vec_list.append(ner_logits_cls[batch_idx][item_idx])
            chosen_vec_tensor = torch.stack(chosen_vec_list, 0)
            new_enc_output = torch.cat([ner_logits_cls[batch_idx], chosen_vec_tensor])
            pad_len = self.config.max_position_embeddings - new_enc_output.size()[0]
            new_enc_output = torch.cat([new_enc_output, ner_logits_cls[batch_idx][-1,:].repeat(pad_len, 1).to(input_ids.device)])

            new_enc_outputs.append(new_enc_output)
        new_hidden_states = torch.stack(new_enc_outputs, 0)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=new_hidden_states, hidden_states=ner_logits, attentions=all_attentions
        )


class BartModel_implicit(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = Bart_imp_Encoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.summary = Summary(emb_dim=config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class BartEncDec_NER_implicit(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn_model\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.summary = Summary(emb_dim=config.d_model)
        self.max_position = config.max_position_embeddings
        self.metric = load_metric("seqeval")
        self.init_weights()
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.pad_token_id = config.pad_token_id
        self.model = BartModel_implicit(config)

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            decoder_input_ids=None,
            labels=None,
            ner_labels=None,
            encoder_outputs=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,

    ):

        if inputs_embeds == None:
            outputs = self.model(input_ids=input_ids, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)
        else:
            outputs = self.model(inputs_embeds=inputs_embeds, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)

        ner_logits = outputs['encoder_hidden_states']  # batch, encseqlen, dim last_encoder_hidden_states_ori

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # batch, decseqlen, dim

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        ner_loss = None
        ner_results = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ner_loss = loss_fct(ner_logits.view(-1, 3), ner_labels.view(-1).long())
            predictions = torch.argmax(ner_logits, dim=2)
            
            true_predictions = [
                [self.id2label[p.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            true_labels = [
                [self.id2label[l.item()] for (p, l) in zip(prediction, label) if l != -1]
                for prediction, label in zip(predictions, ner_labels)
            ]
            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            ner_results = {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

        return ModelOutput(
            logits=lm_logits,
            lm_loss=masked_lm_loss,
            ner_logits=ner_logits,
            ner_loss=ner_loss,
            ner_results=ner_results
        )