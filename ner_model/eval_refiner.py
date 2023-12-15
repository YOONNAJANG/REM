from setproctitle import setproctitle
setproctitle("leejeongwoo")

import os, json
import logging
from argparse import ArgumentParser
print(os.getcwd())

import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from data_utils_refine import add_special_tokens_test, special_tokens_focus, dataloader_focus_test, dataloader_wow_test, add_special_tokens_, dataloader_cmudog_test, dataloader_chatgpt_test
from datasets import load_metric
import re
from tqdm import tqdm


from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
from metrics.dae_factuality.evaluate_factuality import score_example_single_context
from metrics.distinctN import distinct_n_sentence_level

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

logger = logging.getLogger(__file__)

modified = 0

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.do_sample = self.hparams.do_sample
        self.num_beams = self.hparams.num_beams
        self.num_return_sequences = self.hparams.num_return_sequences
        self.top_k = self.hparams.top_k
        self.max_length = self.hparams.max_length
        self.min_length = self.hparams.min_length
        self.no_repeat_ngram_size = self.hparams.no_repeat_ngram_size
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.metric = load_metric("seqeval")

        self.pseudo_token = self.hparams.pseudo_token

        print("Load DAE model weights")
        from transformers import ElectraConfig, ElectraTokenizer
        from metrics.dae_factuality.utils import ElectraDAEModel
        dae_config_class, dae_model_class, dae_tokenizer_class = ElectraConfig, ElectraDAEModel, ElectraTokenizer
        self.dae_tokenizer = dae_tokenizer_class.from_pretrained(self.hparams.dae_model)
        self.dae_model = dae_model_class.from_pretrained(self.hparams.dae_model)
        self.dae_model.to(self.hparams.device)
        
        
        # ROUGE
        self.rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        

        from transformers import AutoTokenizer, BartConfig, BartTokenizer
        from transformers import BartForConditionalGeneration
        if self.hparams.mode == "gen_exp":
            if "bart" in self.hparams.pretrained_model:
                from refiner_modules import BartEncDec_NER_explicit as model
                from transformers import BartConfig as config
                from transformers import BartForConditionalGeneration as congenmodel
            else:
                from refiner_modules import T5EncDec_NER_explicit as model
                from transformers import T5Config as config
                from transformers import T5ForConditionalGeneration as congenmodel
        elif self.hparams.mode == "gen_imp":
            if "bart" in self.hparams.pretrained_model:
                from refiner_modules import BartEncDec_NER_implicit as model
                from transformers import BartConfig as config
                from transformers import BartForConditionalGeneration as congenmodel
            else:
                from refiner_modules import T5EncDec_NER_implicit as model #NotImplemented
                from transformers import T5Config as config
                from transformers import T5ForConditionalGeneration as congenmodel
        elif self.hparams.mode == "original":
            if "bart" in self.hparams.pretrained_model:
                from transformers import BartForConditionalGeneration as model
                from transformers import BartForConditionalGeneration as congenmodel
            else:
                from transformers import T5ForConditionalGeneration as model
                from transformers import T5ForConditionalGeneration as congenmodel
        elif self.hparams.mode == "ner":
            if "bart" in self.hparams.pretrained_model:
                from refiner_modules import BartEncDec as model
                from transformers import BartConfig as config
                from transformers import BartForConditionalGeneration as congenmodel
            else:
                from refiner_modules import T5EncDec as model
                from transformers import T5Config as config
                from transformers import T5ForConditionalGeneration as congenmodel
        else:
            raise NotImplementedError

        self.config = config.from_pretrained(self.hparams.pretrained_model)
        self.model = model.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.congenmodel = congenmodel.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer, self.model, self.congenmodel = add_special_tokens_test(self.model, self.congenmodel, self.tokenizer, special_tokens=special_tokens_focus)

        print('hparams: ', self.hparams)
        print('ptuning: ', self.hparams.ptuning)
        print('mode: ', self.hparams.mode)
        # if self.hparams.ptuning==True:

            # self.tokenizer, self.model, self.congenmodel = add_special_tokens_test(self.model, self.congenmodel,
            #                                                                        self.tokenizer,
            #                                                                        special_tokens={'pseudo_token':self.pseudo_token})
            # for name, param in self.model.named_parameters():
            #     # print('not frozen params: ', name)
            #     # if name.startswith('model.encoder.'):
            #     param.requires_grad = False
            # self.embeddings = get_embedding_layer(self.hparams, self.model)
            # # set allowed vocab set
            # self.vocab = self.tokenizer.get_vocab()
            # self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.hparams, self.tokenizer))
            # self.template = tuple([int(item) for item in self.hparams.template.split(',')])
            # # load prompt encoder
            # self.hidden_size = self.embeddings.embedding_dim
            # self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pseudo_token]})
            #
            # self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
            # self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            # self.spell_length = sum(self.template)
            # self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.hparams.device, self.hparams)
            # self.prompt_encoder = self.prompt_encoder.to(self.hparams.device)

        self.model.to(self.hparams.device)
        self.congenmodel.to(self.hparams.device)
        self.model.eval()
        self.congenmodel.eval()
        self.human_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.human_token])[0]


        if len(self.hparams.checkpoint) > 0:
            checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
            checkpoint_congen = {k[6:]: v for k, v in checkpoint.items()}
            self.congenmodel.load_state_dict(checkpoint_congen, strict=False)
            self.checkpoint_loaded = dict()
            self.checkpoint_prompt = dict()
            for k, v in checkpoint.items():
                if k.startswith('model.'):
                    self.checkpoint_loaded[k[6:]] = v
                else:
                    self.checkpoint_prompt[k] = v
            self.model.load_state_dict(self.checkpoint_loaded, strict=False)
            # self.congenmodel.load_state_dict(checkpoint, strict=False)

    def embed_inputs(self, queries):
        bz = queries.shape[0] #batchsize
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #bsz, seqlen, embdim
        blocked_indices = (queries == self.pseudo_token_id)
        blocked_indices = torch.nonzero(blocked_indices, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # True index tensors -> bz, spell_length, 2 -> :,:,1 (한 입력마다 해당 인덱스 불러옴) ->bsz, spell_length
        replace_embeds = self.prompt_encoder() #spell_length, embdim
        self.prompt_encoder.load_state_dict(self.checkpoint_prompt, strict=False)
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :] #해당 토큰 자리만 replace embedding
        return raw_embeds


    def step(self, batch, batch_idx):
        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(batch['input_ids'])
            if 'input_ids' in batch:
                batch['inputs_embeds'] = input_embeds
            output = self.model(**batch)
        else:
            output = self.model(**batch)
        return output


    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch

        reply_mask = (lm_labels != -100)
        reply = lm_labels[reply_mask]

        input_mask = (input_ids != self.tokenizer.pad_token_id)
        input_ids = input_ids[input_mask].unsqueeze(0)
        ner_labels = ner_labels[input_mask].unsqueeze(0)

        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids,
            'labels':lm_labels,
            'ner_labels':ner_labels
        }
        results = self.step(inputs, batch_idx)
        # print(results.items()) # ner_logits, ner_loss, lm_logits, lm_loss, ner_results
        
        
        # refine 할지말지 결정 ####################################################################################################
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        if 'bart' in self.hparams.pretrained_model:
            if self.hparams.data_type == "focus":
                knowledge_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.knowledge_token)
                knowledge_sp_idx = (input_ids == knowledge_sp_id).nonzero(as_tuple=True)[1][0]
                persona_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.persona_token)
                persona_sp_idx = (input_ids == persona_sp_id).nonzero(as_tuple=True)[1][0]
                knowledge = input_ids[:, knowledge_sp_idx + 1:persona_sp_idx]
                knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=False)
            else:
                knowledge_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.knowledge_token)
                knowledge_sp_idx = (input_ids == knowledge_sp_id).nonzero(as_tuple=True)[1][0]
                human_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.human_token)
                human_sp_idx = (input_ids == human_sp_id).nonzero(as_tuple=True)[1][0]
                knowledge = input_ids[:, knowledge_sp_idx + 1:human_sp_idx]
                knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=False)

        else:
            if self.hparams.data_type == "focus":
                colon_id = self.tokenizer.convert_tokens_to_ids(':')
                colon_idx = (input_ids == colon_id).nonzero(as_tuple=True)[1][0]
                persona_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.persona_token)
                persona_sp_idx = (input_ids == persona_sp_id).nonzero(as_tuple=True)[1][0]
                knowledge = input_ids[:, colon_idx + 1:persona_sp_idx]
                knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=False)
            elif self.hparams.data_type == "wow":
                colon_id = self.tokenizer.convert_tokens_to_ids(':')
                colon_idx = (input_ids == colon_id).nonzero(as_tuple=True)[1][0]
                human_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.human_token)
                human_sp_idx = (input_ids == human_sp_id).nonzero(as_tuple=True)[1][0]
                knowledge = input_ids[:, colon_idx + 1:human_sp_idx]
                knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=False)
            else:#cmu
                knowledge_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.knowledge_token)
                knowledge_sp_idx = (input_ids == knowledge_sp_id).nonzero(as_tuple=True)[1][0]
                human_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.human_token)
                human_sp_idx = (input_ids == human_sp_id).nonzero(as_tuple=True)[1][0]
                knowledge = input_ids[:, knowledge_sp_idx + 1:human_sp_idx]
                knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=False)
        
        before_refine = input_text[0].split("<human>")[-1]

        # ROUGE-L
        input_kg_rougeL = self.rouge_metric.score(before_refine, knowledge)['rougeL'].fmeasure
        
        # DAE
        clean_before_refine = re.sub("[^\w|\s]", "", before_refine, 0, re.IGNORECASE)
        clean_before_refine = clean_before_refine.strip()
        
        clean_knowledge = re.sub("[^\w|\s}]", "", knowledge, 0, re.IGNORECASE)
        clean_knowledge = clean_knowledge.strip()
        
        clean_knowledge = self.dae_tokenizer.decode(self.dae_tokenizer(clean_knowledge)['input_ids'][:120-len(self.dae_tokenizer(clean_before_refine)['input_ids'])], skip_special_tokens=True)
        if len(clean_before_refine) == 0:
            input_kg_dae = 0
        else:
            input_kg_dae = score_example_single_context(clean_before_refine, clean_knowledge, self.dae_model, self.dae_tokenizer, self.hparams)
        input_kg_dae = float(input_kg_dae)

        

        ## ROUGE-L과 DAE로 refine 할지말지 결정
        if input_kg_dae < self.hparams.refine_threshold:
            is_refine = True
            global modified
            modified += 1
        else:
            is_refine = False

        # print("before_refine:", before_refine)
        # print("knowledge:", knowledge)
        # print("input_kg_rougeL:", input_kg_rougeL)
        # print("input_kg_dae:", input_kg_dae)
        # print()
        
        ########################################################################################################################
        
        if is_refine is False:
            ppl = torch.exp(results["loss"])
            
            out_ids = self.tokenizer([before_refine])['input_ids']
            out_ids = torch.cuda.IntTensor(out_ids)
            
            # refine 하지 않을 때 ner_results
            ner_results = None
        else:
            if self.hparams.mode == "original":
                lm_logits = results['logits']
                ppl = torch.exp(results["loss"])

                with torch.no_grad():
                    out_ids = self.congenmodel.generate(input_ids=input_ids,
                                                do_sample=self.do_sample, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences,
                                                top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size,
                                                min_length=self.min_length, max_length=self.max_length)
                ner_result = None

            else:
                # print(results.keys())
                lm_logits, ner_logits = results['logits'], results['ner_logits']
                ppl = torch.exp(results["loss"])

                result = {}
                for k, v in results.items():
                    if k != "ner_results":
                        result[k] = v.detach().cpu()
                    else:
                        result[k] = v
                predictions = torch.argmax(ner_logits, dim=-1)
                pred_all = (predictions == 1) + (predictions == 2)

                if self.hparams.mode == "gen_exp":

                    chosen_tok_list = []
                    for batch_index, batch_item in enumerate(pred_all):
                        for item_idx, item in enumerate(batch_item):
                            if item == True:
                                chosen_tok_list.append(input_ids[batch_index][item_idx])
                        chosen_tok_list = [torch.tensor(2).to(input_ids.device)] + chosen_tok_list + [torch.tensor(2).to(input_ids.device)]
                        new_dec_input = torch.stack(chosen_tok_list, 0).unsqueeze(0)

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
                    "accuracy": results["overall_accuracy"]}




            if self.hparams.mode == "gen_exp":
                with torch.no_grad():
                    out_ids = self.congenmodel.generate(input_ids=input_ids, decoder_input_ids=new_dec_input,
                                                do_sample=self.do_sample, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences,
                                                top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size,
                                                min_length=self.min_length, max_length=self.max_length)

                if self.num_return_sequences > 1:
                    new_out_ids = []
                    for out_id in out_ids:
                        output_index = (out_id == 2).nonzero(as_tuple=True)[0][1].item()
                        new_out_ids.append(out_id[output_index:])
                    out_ids = torch.stack(new_out_ids, 0)
                else:
                    output_index = (out_ids[0] == 2).nonzero(as_tuple=True)[0][1].item()
                    out_ids = out_ids[0][output_index:].unsqueeze(0)

            elif self.hparams.mode == "gen_imp":
                with torch.no_grad():
                    out_ids = self.congenmodel.generate(input_ids=input_ids,
                                                do_sample=self.do_sample, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences,
                                                top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size,
                                                min_length=self.min_length, max_length=self.max_length)


            elif self.hparams.mode == "ner":
                with torch.no_grad():
                    out_ids = self.congenmodel.generate(input_ids=input_ids,
                                                do_sample=self.do_sample, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences,
                                                top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size,
                                                min_length=self.min_length, max_length=self.max_length)

            else:
                raise NotImplementedError

        
        reply = self.tokenizer.decode(reply.tolist(), skip_special_tokens=True)
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        out_ids = out_ids[:, :self.hparams.max_length]
        out_ids = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        out_ids = [out_id[:self.hparams.max_length] for out_id in out_ids]
        only_input = input_ids.tolist()[0]
        human_idx = only_input.index(self.human_token_id)
        only_input = only_input[human_idx:]
        only_input = self.tokenizer.decode(only_input, skip_special_tokens=True)


        knoweldge_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.knowledge_token)
        try:
            knoweldge_sp_idx = (input_ids == knoweldge_sp_id).nonzero(as_tuple=True)[1][0]
        except IndexError:
            knoweldge_sp_id = self.tokenizer.convert_tokens_to_ids(':')
            knoweldge_sp_idx = (input_ids == knoweldge_sp_id).nonzero(as_tuple=True)[1][0]

        if self.hparams.data_type == "focus":

            persona_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.persona_token)
            persona_sp_idx = (input_ids == persona_sp_id).nonzero(as_tuple=True)[1][0]

            knowledge = input_ids[:, knoweldge_sp_idx + 1:persona_sp_idx]
            knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=True)
        else:
            human_sp_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.human_token)
            human_sp_idx = (input_ids == human_sp_id).nonzero(as_tuple=True)[1][0]

            knowledge = input_ids[:, knoweldge_sp_idx + 1:human_sp_idx]

            knowledge = self.tokenizer.decode(knowledge.squeeze(0).tolist(), skip_special_tokens=True)


        # print('input: ', input_text, '\n true: ', reply, '\n pred: ', out_ids)
        result = dict()
        # print('ppl: ', ppl)
        result['ppl'] = ppl
        result['y_true_text'] = reply  # tokenize!!!
        result['y_pred_text'] = out_ids
        result['input_text'] = input_text
        result['only_input'] = only_input
        result['ner_results'] = ner_results
        result['knowledge'] = knowledge
        result['refine'] = str(is_refine)

        return result

    def epoch_end(self, outputs, state='test'):

        text_result = []
        for index, i in enumerate(outputs):
            text_dict = dict()
            text_dict['ppl'] = i['ppl']
            text_dict['y_true_text'] = i['y_true_text']
            text_dict['y_pred_text'] = i['y_pred_text']
            text_dict['input_text'] = i['input_text']
            text_dict['knowledge'] = i['knowledge']
            text_dict['refine'] = i['refine']
            text_dict['only_input'] = i['only_input']
            if i['ner_results'] is not None:
                text_dict['ner_results'] = i['ner_results']

            # text_dict['model_pred_knowledge'] = i['model_pred_knowledge']

            text_result.append(text_dict)

        result = {'text_result': text_result}
        return result

    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')

        print("Load NER tagger")
        from flair.data import Sentence
        from flair.models import SequenceTagger
        tagger = SequenceTagger.load("flair/ner-english-large")

        text_result = result['text_result']
        ner_acc = 0
        ner_rec = 0
        ner_prec = 0
        ner_f1 = 0
        ppl = 0
        r1 = 0
        r2 = 0
        rl = 0
        bleu = 0
        chrf = 0
        dae = 0
        dist1 = 0
        dist2 = 0
        tc = 0
        ec = 0
        k_overlap = 0
        bleu_metric = load_metric("sacrebleu")
        chrf_metric = CHRFScore()

        result_list = list()

        for test_data_index, test_data in enumerate(tqdm(text_result)):
            pred_dict = dict()
            ppl += test_data['ppl']
            # print('ppl accumulated: ', ppl)
            gold_reply = test_data['y_true_text']
            pred_reply = test_data['y_pred_text']
            input = test_data['input_text']
            refine = test_data['refine']
            only_input = test_data['only_input']

            if self.hparams.only_score_refine == True and refine == 'False':
                continue

            if 'ner_results' in test_data.keys():

                ner_acc += test_data['ner_results']['accuracy']
                ner_rec += test_data['ner_results']['recall']
                ner_prec += test_data['ner_results']['precision']
                ner_f1 += test_data['ner_results']['f1']

            knowledge = test_data['knowledge']

            pred_dict['input'] = input
            pred_dict['gold'] = gold_reply
            pred_dict['pred'] = pred_reply
            pred_dict['knoweldge'] = knowledge
            pred_dict['refine'] = refine
            pred_dict['only_input'] = only_input

            result_list.append(pred_dict)


            # ROUGE
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    r = self.rouge_metric.score(pred_reply_item, gold_reply)
                    r1 += r['rouge1'].fmeasure
                    r2 += r['rouge2'].fmeasure
                    rl += r['rougeL'].fmeasure
            else:
                r = self.rouge_metric.score(pred_reply[0], gold_reply)
                r1 += r['rouge1'].fmeasure
                r2 += r['rouge2'].fmeasure
                rl += r['rougeL'].fmeasure

            # sacre BLEU
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    bleu += bleu_metric.compute(predictions=[pred_reply_item], references=[[gold_reply]])['score']
            else:
                bleu += bleu_metric.compute(predictions=pred_reply, references=[[gold_reply]])['score']

            # knowledge overlapping (knowledge-rouge-L)
            # if self.hparams.num_return_sequences > 1:
            #     for pred_reply_item in pred_reply:
            #         r = self.rouge_metric.score(pred_reply_item, gold_reply)
            #         k_overlap += r['rougeL'].fmeasure
            # else:
            #     r = self.rouge_metric.score(pred_reply[0], gold_reply)
            #     k_overlap += r['rougeL'].fmeasure

            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    k_overlap += bleu_metric.compute(predictions=[pred_reply_item], references=[[gold_reply]])['score']
            else:
                k_overlap += bleu_metric.compute(predictions=pred_reply, references=[[gold_reply]])['score']

            # ChrF++
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply_item, 0, re.IGNORECASE)
                    chrf += chrf_metric([pred_reply_wo_specialchar], [[gold_reply]]).clone().detach()
            else:
                pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply[0], 0, re.IGNORECASE)
                chrf += chrf_metric([pred_reply_wo_specialchar], [[gold_reply]]).clone().detach()


            # print('dae')
            # dae_factuality
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply_item, 0, re.IGNORECASE)
                    print(pred_reply_wo_specialchar, len(pred_reply_wo_specialchar), type(pred_reply_wo_specialchar))
                    knowledge_wo_specialchar = re.sub("[^\w|\s]", "", knowledge, 0, re.IGNORECASE)
                    knowledge_wo_specialchar = knowledge_wo_specialchar.strip()
                    if len(pred_reply_wo_specialchar) == 0:
                        dae += 0
                    else:
                        dae += score_example_single_context(pred_reply_wo_specialchar, knowledge_wo_specialchar, self.dae_model, self.dae_tokenizer,
                                                        self.hparams)

            else:
                pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply[0], 0, re.IGNORECASE)
                pred_reply_wo_specialchar = pred_reply_wo_specialchar.strip()
                print(pred_reply_wo_specialchar, len(pred_reply_wo_specialchar), type(pred_reply_wo_specialchar))
                knowledge_wo_specialchar = re.sub("[^\w|\s}]", "", knowledge, 0, re.IGNORECASE)
                knowledge_wo_specialchar = knowledge_wo_specialchar.strip()
                if len(pred_reply_wo_specialchar) == 0 :
                    dae += 0
                else:
                    dae += score_example_single_context(pred_reply_wo_specialchar, knowledge, self.dae_model, self.dae_tokenizer, self.hparams)
            # print('dae_score', dae)

            # print('distN')
            # distinct-N
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    dist1 += distinct_n_sentence_level(pred_reply_item, 1)
                    dist2 += distinct_n_sentence_level(pred_reply_item, 2)
            else:
                dist1 += distinct_n_sentence_level(pred_reply[0], 1)
                dist2 += distinct_n_sentence_level(pred_reply[0], 2)


            # print("TC")
            pred_format = {'LOC': {"keyword": []},
                           'MISC': {"keyword": []},
                           'PER': {"keyword": []},
                           'ORG': {"keyword": []},
                           }
            gold_format = {'LOC': {"keyword": []},
                           'MISC': {"keyword": []},
                           'PER': {"keyword": []},
                           'ORG': {"keyword": []},
                           }
            knowledge_format = {'LOC': {"keyword": []},
                           'MISC': {"keyword": []},
                           'PER': {"keyword": []},
                           'ORG': {"keyword": []},
                           }
            tmp_ec = 0
            tmp_tc = 0
            if pred_reply[0] != "":

                #pred_reply, gold_reply
                sentence = pred_reply[0]
                sentence = re.sub("[^\w|\s]", "", sentence, 0, re.IGNORECASE)

                sentence = Sentence(sentence)
                tagger.predict(sentence)
                for entity in sentence.get_spans('ner'):
                    if len(pred_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                            pred_format[entity.get_label("ner").value]["keyword"]:
                        # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                        pred_format[entity.get_label("ner").value]["keyword"].append(entity.text)
                sentence = Sentence(gold_reply)
                tagger.predict(sentence)
                for entity in sentence.get_spans('ner'):
                    if len(gold_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                            gold_format[entity.get_label("ner").value]["keyword"]:
                        # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                        gold_format[entity.get_label("ner").value]["keyword"].append(entity.text)
                sentence = Sentence(knowledge)
                tagger.predict(sentence)
                for entity in sentence.get_spans('ner'):
                    if len(knowledge_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                            knowledge_format[entity.get_label("ner").value]["keyword"]:
                        # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
                        knowledge_format[entity.get_label("ner").value]["keyword"].append(entity.text)


                for key in gold_format.keys():
                    gold_w_num = len(gold_format[key]["keyword"])
                    pred_w_num = len(pred_format[key]["keyword"])
                    if gold_w_num == 0:
                        continue
                    tc_ratio = pred_w_num / gold_w_num
                    tmp_tc += tc_ratio
                tmp_tc = tmp_tc /4


                # print("EC")
                pred_k_list = []
                gold_k_list = []
                knowledge_k_list = []
                for key in gold_format.keys():
                    pred_k_list.extend(pred_format[key]["keyword"])
                    gold_k_list.extend(gold_format[key]["keyword"])
                    knowledge_k_list.extend(knowledge_format[key]["keyword"])

                knowledge_gold = list(set(knowledge_k_list) & set(gold_k_list))
                # print(knowledge_gold)
                knowledge_gold_pred = list(set(knowledge_gold) & set(pred_k_list))
                # print(knowledge_gold_pred)
                if len(knowledge_gold) == 0:
                    tmp_ec = 0
                else:
                    tmp_ec = len(knowledge_gold_pred) / len(knowledge_gold)

            tc += tmp_tc
            ec += tmp_ec

        if self.hparams.only_score_refine == True:
            test_data_index = modified - 1
            print('only refine')


        chrf_result = chrf / ((test_data_index + 1) * self.hparams.num_return_sequences)
        rouge1_result = r1 / ((test_data_index + 1) * self.hparams.num_return_sequences)
        rouge2_result = r2 / ((test_data_index + 1) * self.hparams.num_return_sequences)
        rougel_result = rl / ((test_data_index + 1) * self.hparams.num_return_sequences)
        bleu_result = bleu / ((test_data_index + 1) * self.hparams.num_return_sequences)
        ppl_result = ppl / (test_data_index + 1)
        dae_result = dae / ((test_data_index + 1) * self.hparams.num_return_sequences)
        dist1_result = dist1 / ((test_data_index + 1) * self.hparams.num_return_sequences)
        dist2_result = dist2 / ((test_data_index + 1) * self.hparams.num_return_sequences)
        tc_result = tc / (test_data_index + 1)
        ec_result = ec / (test_data_index + 1)
        k_overlap_result = k_overlap / ((test_data_index + 1) * self.hparams.num_return_sequences)


        if self.hparams.mode != "original":
            ner_acc_result = ner_acc / (test_data_index + 1)
            ner_rec_result = ner_rec / (test_data_index + 1)
            ner_prec_result = ner_prec / (test_data_index + 1)
            ner_f1_result = ner_f1 / (test_data_index + 1)


        result_dict = dict()
        result_dict['chrF++'] = chrf_result.item()
        result_dict['rouge1'] = rouge1_result
        result_dict['rouge2'] = rouge2_result
        result_dict['rougeL'] = rougel_result
        result_dict['bleu'] = bleu_result
        result_dict['ppl'] = ppl_result.item()
        result_dict['tc'] = tc_result
        result_dict['ec'] = ec_result
        result_dict['dae_result'] = dae_result
        result_dict['dist1_result'] = dist1_result
        result_dict['dist2_result'] = dist2_result
        result_dict['k_overlap'] = k_overlap_result

        if self.hparams.mode != "original":
            result_dict['ner_acc'] = ner_acc_result
            result_dict['ner_rec'] = ner_rec_result
            result_dict['ner_prec'] = ner_prec_result
            result_dict['ner_f1'] = ner_f1_result

        print(result_dict.items())

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list

        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')

        print(f"{modified} out of {test_data_index+1} has been modified")

        return self.epoch_end(outputs, state='test')

    def dataloader(self):
        test_dataset = None

        if self.hparams.data_type == "focus":
            test_dataset = dataloader_focus_test(self.hparams, self.tokenizer, self.hparams.test_dataset_path,
                                                 self.hparams.test_dataset_cache)
        elif self.hparams.data_type == "wow":
            test_dataset = dataloader_wow_test(self.hparams, self.tokenizer, self.hparams.test_dataset_path,
                                               self.hparams.test_dataset_cache)
        elif self.hparams.data_type == "cmudog":
            test_dataset = dataloader_cmudog_test(self.hparams, self.tokenizer, self.hparams.test_dataset_path,
                                               self.hparams.test_dataset_cache)
        elif self.hparams.data_type == "chatgpt":
            test_dataset = dataloader_chatgpt_test(self.hparams, self.tokenizer, self.hparams.test_dataset_path,
                                               self.hparams.test_dataset_cache)

        return test_dataset

    def test_dataloader(self):
        test_dataset = self.dataloader()
        print("Valid dataset (Batch, Seq length): {}".format(test_dataset.tensors[0].shape))
        return DataLoader(test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)

def main():

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, cmudog, chatgpt}")
    parser.add_argument("--test_dataset_path", type=str, default="/home/data/ssh5131/FoCus_data/our_data/test_ours.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str,
                        default='/home/data/ssh5131/FoCus_data/our_data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}"") #facebook/bart-base")
    parser.add_argument("--mode", type=str, default="gen_imp", help="{ner, gen_exp, gen_imp, original}")
    parser.add_argument("--ckpt", type=str, default="facebook/bart-base", help="ckpt path") #facebook/bart-base
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--random_seed", type=int, default=644128)
    parser.add_argument("--precision", type=int, default=32, help="{16,32,64}")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=512, help="maximum length")
    parser.add_argument("--min_length", type=int, default=32, help="minimum length")
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="{1, 2, 5, 10}, 1 for 1 generated result")
    parser.add_argument("--output_dir", type=str, default="/home/data/yoonna/focus_modeling/eval_output/focus_refiner/", help="default value for PLMs")
    parser.add_argument("--dae_model", type=str, default="/home/data/yoonna/Refiner/metrics/dae_factuality/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    parser.add_argument("--seed", type=int, default=19981014, help="Seed")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--knowledge_select", type=str, default="None", help="{None, DPR, BM25, TFIDF}")
    parser.add_argument("--refine_threshold", type=float, default=0.0, help="0<=threshold<=1")
    parser.add_argument("--only_score_refine", type=bool, default=False)


    #for p-tuning
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="50,50,50") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])


    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])



    torch.manual_seed(args['seed'])
    seed_everything(args['seed'], workers=True)

    if args['gpu_num'] == 1:
        args['distributed'] = False
    elif args['gpu_num'] > 1:
        args['distributed'] = True
    else:
        raise NotImplementedError

    print(":: Prepare tokenizer and pretrained model ::")
    model = Model(**args)
    model.to(args['device'])

    flag = args['flag']

    trainer_args = {
        'num_sanity_val_steps': 2,  # None if args['test_mode'] else 0
        'deterministic': False,
        'gpus': args['gpu_num'],
        'strategy': DDPPlugin(find_unused_parameters=True),
        'precision': 32}

    print(":: Start Testing ::")
    trainer = Trainer(**trainer_args)

    model.freeze()
    with torch.no_grad():
        trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    main()

