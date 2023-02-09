#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from argparse import ArgumentParser
import wandb, json
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import Sigmoid, Softmax, CrossEntropyLoss
from data_utils import get_testdata_loaders, add_special_tokens_test, special_tokens
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from sklearn.metrics import accuracy_score
from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
from cusgen_generate import generate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from dae_factuality.evaluate_factuality import score_example_single_context
from distinctN import distinct_n_sentence_level


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer
            from cusgen_generate import Bart_pkgen as bartmodel
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = bartmodel.from_pretrained(self.hparams.model_path)

            from transformers import BartForConditionalGeneration
            self.congenmodel = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.tokenizer, self.model, self.congenmodel = add_special_tokens_test(self.model, self.congenmodel, self.tokenizer, special_tokens)

            self.model.to(self.hparams.device)
            self.congenmodel.to(self.hparams.device)
            self.model.eval()
            self.congenmodel.eval()


            if len(self.hparams.checkpoint) > 0:
                checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
                checkpoint = {k[6:]: v for k, v in checkpoint.items()}
                self.model.load_state_dict(checkpoint)
                self.congenmodel.load_state_dict(checkpoint, strict=False)


        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer
            from cusgen_generate import T5_pkgen as t5model
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = t5model.from_pretrained(self.hparams.model_path)

            from transformers import T5ForConditionalGeneration
            self.congenmodel = T5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.tokenizer, self.model, self.congenmodel = add_special_tokens_test(self.model, self.congenmodel,
                                                                                   self.tokenizer, special_tokens)

            self.model.to(self.hparams.device)
            self.congenmodel.to(self.hparams.device)
            self.model.eval()
            self.congenmodel.eval()

            if len(self.hparams.checkpoint) > 0:
                checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
                checkpoint = {k[6:]: v for k, v in checkpoint.items()}
                self.model.load_state_dict(checkpoint)
                self.congenmodel.load_state_dict(checkpoint, strict=False)
            # self.tokenizer, self.model = add_special_tokens_test(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'LED':
            from transformers import LEDTokenizer
            from cusgen_generate import LED_pkgen as ledmodel
            self.tokenizer = LEDTokenizer.from_pretrained(self.hparams.model_path)
            self.model = ledmodel.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'transformer-encdec':
            from transformers import BartTokenizer, BartConfig
            from cusgen_generate import BARTPK_ctxt as bartmodel
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model_config = BartConfig.from_pretrained(self.hparams.model_path)
            self.model = bartmodel(self.model_config)
            self.model.to(self.hparams.device)
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        else:
            raise NotImplementedError

        test_dataset = get_testdata_loaders(self.hparams, self.tokenizer)
        self.test_dataset = test_dataset

    def test_dataloader(self):
        print("\n::: Load and preprocess TEST dataset :::")
        # test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size, num_workers=3)
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.hparams.test_batch_size)
        return test_loader


    def step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(**batch)
        result = {'lm_logits':output['dynamic_lm_logits'], 'knowledge_logits':output['knowledge_logits'], 'persona_logits':output['persona_logits']}

        return result

    def test_step(self, batch, batch_idx):
        # input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
        # knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = batch
        # print(len(batch))

        input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, persona_grounding, \
        knowledge_candidates, knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog, history_list = batch


        mask = (reply != self.tokenizer.pad_token_id)
        reply = reply[mask]
        inputs = {
            'input_ids':input_ids,
            'input_eos':input_eos,
            'only_dial_input_ids':dialog,
            'decoder_input_ids':decoder_input_ids,
            'persona_input_ids':persona_candidates,
            'knowledge_input_ids':knowledge_candidates,
            'persona_can_idx':persona_can_idx,
            'knowledge_can_idx':knowledge_can_idx,
            'tot_knowledge':tot_knowledge,
            'tot_knowledge_eos':tot_knowledge_eos,
            'training':False,
            'lm_labels':lm_labels
        }

        result = self.step(inputs, batch_idx)

        lm_logits, knowledge_logits, persona_logits = result['lm_logits'], result['knowledge_logits'], result['persona_logits']


        lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(lm_loss)

        persona, knowledge = self.tokenizer.convert_tokens_to_ids(list(special_tokens.values())[-2:])
        bos, padding, eos = 0, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id  # 0, 1, 2

        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        bos_tensor = torch.tensor([bos]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)
        max_position = 1024

        sigmoid = Sigmoid()
        persona_pred_sigmoid = sigmoid(persona_logits)
        persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
        all_persona_pred = []
        selected_persona_idx = list()
        for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
            batch_list_idx = list()
            batch_list = list()
            for i, can in enumerate(persona_batch):
                if can == True:
                    batch_list_idx.append(can)
                    persona_selected_now = persona_candidates[batch_idx][i]
                    mask_persona = torch.ne(persona_selected_now, padding)
                    persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                    batch_list.append(persona_selected_now[1:-1])
            all_persona_pred.append(batch_list)
            selected_persona_idx.append(batch_list_idx)

        p_index_cvtd = persona_pred_sigmoid[0]
        softmax = Softmax(dim=-1)
        knowledge_softmax = softmax(knowledge_logits)
        _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
        all_knowledge_pred = []

        for batch_i in range(self.hparams.test_batch_size):
            knowledge_pred_idx = k_index_1[batch_i]
            knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
            mask_knowledge = torch.ne(knowledge_pred, padding)
            knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
            knowledge_pred = knowledge_pred[1:-1]
            all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos

        #k_index_1 = k_index_1.squeeze(0)
        #k_index_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=self.hparams.device)

        for batch_i in range(self.hparams.test_batch_size):
            only_dial_input_ids_batch = dialog[batch_i]
            mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
            only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
            if len(all_persona_pred[batch_i])>0:
                concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                new_persona = concat_persona
                #new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
            else:
                new_persona = None

            #new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
            new_knowledge = all_knowledge_pred[batch_i]

            if new_persona is not None:
                new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)
            else:
                new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)

        prev_utt = self.tokenizer.decode(only_dial_input_ids_batch, skip_special_tokens=True)
        all_persona_txt = [self.tokenizer.decode(persona, skip_special_tokens=True) for persona in persona_candidates[0].tolist()]

        with torch.no_grad():
            #out_ids = sample_sequence(new_input.unsqueeze(0), token_type_ids=None, decoder_input_ids=decoder_input_ids, tokenizer=self.tokenizer, model=self.model, args=self.hparams, current_output=None)
            num_beams = self.hparams.num_beams
            num_return_sequences = self.hparams.num_beams
            top_k = self.hparams.top_k
            # input_ids, model, num_beams, num_return_sequences, top_k
            out_ids = generate(new_input.unsqueeze(0), model=self.congenmodel, num_beams=num_beams, num_return_sequences=num_return_sequences,top_k=top_k)
            persona_grounding = persona_grounding.type_as(persona_logits).squeeze()

        reply = self.tokenizer.decode(reply.tolist(), skip_special_tokens=True)

        input_text = self.tokenizer.decode(new_input.tolist(), skip_special_tokens=True)
        model_pred_k = self.tokenizer.decode(new_knowledge.tolist(), skip_special_tokens=True)
        if num_beams > 1:
            out_ids = [self.tokenizer.decode(output_item, skip_special_tokens=True) for output_item in out_ids.tolist()]
        else:
            out_ids = self.tokenizer.decode(out_ids.squeeze(0).tolist(), skip_special_tokens=True)


        #print('input: ', input_text, '\n true: ', reply, '\n pred: ', out_ids, '\n model_k: ', model_pred_k)
        result = dict()
        result['ppl'] = ppl
        result['pg_pred'] = p_index_cvtd
        result['pg_true'] = persona_grounding
        # result['kg_pred'] = knowledge_softmax
        result['kg_pred'] = k_index_1.squeeze(0)
        result['kg_true'] = knowledge_grounding
        result['y_true_text'] = reply #tokenize!!!
        result['y_pred_text'] = out_ids
        result['input_ids'] = new_input.tolist()
        result['input_text'] = input_text
        result['model_pred_knowledge'] = model_pred_k
        result['prev_utterance'] = prev_utt
        result['all_persona'] = all_persona_txt
        # self.log('y_true', result['y_true'])
        return result


    def epoch_end(self, outputs, state='test'):
        text_result = []
        for index, i in enumerate(outputs):
            text_dict = dict()
            text_dict['ppl'] = i['ppl']
            text_dict['pg_pred'] = i['pg_pred']
            text_dict['pg_true'] = i['pg_true']
            text_dict['kg_pred'] = i['kg_pred']
            text_dict['kg_true'] = i['kg_true']
            text_dict['y_true_text'] = i['y_true_text']
            text_dict['y_pred_text'] = i['y_pred_text']
            text_dict['input_text'] = i['input_text']
            text_dict['input_ids'] = i['input_ids']
            text_dict['model_pred_knowledge'] = i['model_pred_knowledge']
            text_dict['prev_utterance'] = i['prev_utterance']
            text_dict['all_persona'] = i['all_persona']

            text_result.append(text_dict)

        result = {'text_result':text_result}


        return result


    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')

        pg_true_list = []
        pg_pred_list = []
        for i in outputs:
            pg_true_list.extend(i['pg_true'].tolist())
            pg_pred_list.extend(i['pg_pred'].tolist())
        # confusion = confusion_matrix(pg_true_list, pg_pred_list)
        # accuracy = accuracy_score(pg_true_list, pg_pred_list)
        # precision = precision_score(pg_true_list, pg_pred_list)
        # recall = recall_score(pg_true_list, pg_pred_list)
        f1 = f1_score(pg_true_list, pg_pred_list)

        # print("Load FactCC model weights")
        # from transformers import BertTokenizer, BertConfig
        # from factcc import BertPointer
        # factcc_config = BertConfig.from_pretrained(self.hparams.factcc_model+'/config.json')
        # factcc_model = BertPointer.from_pretrained(self.hparams.factcc_model+'/pytorch_model.bin', config=factcc_config)
        # factcc_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # factcc_model.to(self.hparams.device)
        # factcc_model.eval()


        print("Load DAE model weights")
        from transformers import ElectraConfig, ElectraTokenizer
        from dae_factuality.utils import ElectraDAEModel
        dae_config_class, dae_model_class, dae_tokenizer_class = ElectraConfig, ElectraDAEModel, ElectraTokenizer
        dae_tokenizer = dae_tokenizer_class.from_pretrained(self.hparams.dae_model)
        dae_model = dae_model_class.from_pretrained(self.hparams.dae_model)
        dae_model.to(self.hparams.device)


        text_result = result['text_result']
        ppl = 0
        r1 = 0
        r2 = 0
        rl = 0
        bleu = 0
        chrf = 0
        #factcc_output = 0
        dae = 0
        dist1 = 0
        dist2 = 0
        rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bleu_metric = load_metric("sacrebleu")
        chrf_metric = CHRFScore()
        pg_score = 0
        kg_score = 0

        result_list = list()

        for test_data_index, test_data in enumerate(text_result):
            pred_dict = dict()
            ppl += test_data['ppl']
            p_index_cvtd = test_data['pg_pred']
            persona_grounding = test_data['pg_true']
            k_index_cvtd = test_data['kg_pred']
            knowledge_grounding = test_data['kg_true']
            gold_reply = test_data['y_true_text']
            pred_reply = test_data['y_pred_text']
            input = test_data['input_text']
            input_ids = test_data['input_ids']

            model_pred_knowledge = test_data['model_pred_knowledge']
            prev_utterance = test_data['prev_utterance']
            all_persona = test_data['all_persona']
            # pred_dict['pg_pred'] = p_index_cvtd.int().tolist()
            # pred_dict['pg_true'] = persona_grounding.int().tolist()
            pred_dict['input'] = input
            pred_dict['input_ids'] = input_ids
            pred_dict['gold'] = gold_reply
            pred_dict['pred'] = pred_reply
            pred_dict['model_pred_knowledge'] = model_pred_knowledge
            pred_dict['prev_utterance'] = prev_utterance
            pred_dict['persona'] = all_persona
            result_list.append(pred_dict)

            #ROUGE
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         r = rouge_metric.score(pred_reply_item, gold_reply)
            #         r1 += r['rouge1'].fmeasure
            #         r2 += r['rouge2'].fmeasure
            #         rl += r['rougeL'].fmeasure
            # else:
            #     r = rouge_metric.score(pred_reㅔly, gold_reply)
            #     r1 += r['rouge1'].fmeasure
            #     r2 += r['rouge2'].fmeasure
            #     rl += r['rougeL'].fmeasure


            # #sacre BLEU
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         bleu += bleu_metric.compute(predictions=[pred_reply_item], references=[[gold_reply]])['score']
            # else:
            #     bleu += bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']

            #ChrF++
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         chrf += chrf_metric([pred_reply_item], [[gold_reply]]).clone().detach()
            #
            # else:
            #     chrf += chrf_metric([pred_reply], [[gold_reply]]).clone().detach()

            # # PG
            # p_label_cvtd = torch.tensor([1 if num in persona_grounding else 0 for num in range(5)], device=self.hparams.device)
            # pg_acc = accuracy_score(p_label_cvtd.cpu(), p_index_cvtd.cpu().squeeze())
            # pg_score += pg_acc
            #
            # # KG
            # kg_acc = accuracy_score(knowledge_grounding.cpu(), k_index_cvtd.cpu())
            # kg_score += kg_acc
            #

            # #FactCC
            # if self.hparams.num_beams > 1:
            #     knowledge_input = factcc_tokenizer.tokenize(model_pred_knowledge)
            #     for pred_reply_item in pred_reply:
            #         generated_input = factcc_tokenizer.tokenize(pred_reply_item)
            #         factcc_input = [factcc_tokenizer.cls_token] + knowledge_input + [factcc_tokenizer.sep_token] + generated_input + [factcc_tokenizer.sep_token]
            #         factcc_input = torch.tensor(factcc_tokenizer.convert_tokens_to_ids(factcc_input)).to(self.hparams.device).unsqueeze(0)
            #         with torch.no_grad():
            #             factcc_output += factcc_model(factcc_input).argmax().item()
            # else:
            #     knowledge_input = factcc_tokenizer.tokenize(model_pred_knowledge)
            #     generated_input = factcc_tokenizer.tokenize(pred_reply)
            #     factcc_input = [factcc_tokenizer.cls_token] + knowledge_input + [factcc_tokenizer.sep_token] + generated_input + [factcc_tokenizer.sep_token]
            #     factcc_input = torch.tensor(factcc_tokenizer.convert_tokens_to_ids(factcc_input)).to(self.hparams.device).unsqueeze(0)
            #     with torch.no_grad():
            #         factcc_output += factcc_model(factcc_input).argmax().item()

            # print('knowledge: ', model_pred_knowledge, 'gold reply: ', gold_reply, 'pred reply: ', pred_reply)
            # #dae_factuality
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         dae += score_example_single_context(pred_reply_item, model_pred_knowledge, dae_model, dae_tokenizer, self.hparams)
            # else:
            #     dae += score_example_single_context(pred_reply, model_pred_knowledge, dae_model, dae_tokenizer, self.hparams)
            # # print('dae_score', dae)
            #

            # #distinct-N
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         dist1 += distinct_n_sentence_level(pred_reply_item, 1)
            #         dist2 += distinct_n_sentence_level(pred_reply_item, 2)
            # else:
            #     dist1 += distinct_n_sentence_level(pred_reply, 1)
            #     dist2 += distinct_n_sentence_level(pred_reply, 2)

        # chrf_result = chrf/((test_data_index+1)*self.hparams.num_beams)
        # rouge1_result = r1/((test_data_index+1)*self.hparams.num_beams)
        # rouge2_result = r2/((test_data_index+1)*self.hparams.num_beams)
        # rougel_result = rl/((test_data_index+1)*self.hparams.num_beams)
        # bleu_result = bleu/((test_data_index+1)*self.hparams.num_beams)
        # pg_result = pg_score/(test_data_index+1)
        # kg_result = kg_score/(test_data_index+1)
        # ppl_result = ppl/(test_data_index+1)
        # #factcc_result = factcc_output/((test_data_index+1)*self.hparams.num_beams)
        # dae_result = dae/((test_data_index+1)*self.hparams.num_beams)
        # dist1_result = dist1/((test_data_index+1)*self.hparams.num_beams)
        # dist2_result = dist2/((test_data_index+1)*self.hparams.num_beams)

        result_dict = dict()
        # result_dict['chrF++'] = chrf_result.item()
        # result_dict['rouge1'] = rouge1_result
        # result_dict['rouge2'] = rouge2_result
        # result_dict['rougeL'] = rougel_result
        # result_dict['bleu'] = bleu_result
        # result_dict['pg_acc'] = pg_result
        # result_dict['kg_acc'] = kg_result
        # result_dict['ppl'] = ppl_result.item()
        # result_dict['pg_f1'] = f1
        # #result_dict['factcc_result'] = factcc_result
        # result_dict['dae_result'] = dae_result
        # result_dict['dist1_result'] = dist1_result
        # result_dict['dist2_result'] = dist2_result

        # print(result_dict.items())

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list

        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')
        return test_result



def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="/home/data/ssh5131/focus_modeling/our_data/toy_train_ours.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='/home/data/ssh5131/focus_modeling/our_data/toy_focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/eval_output/", help="Path for the output file to be saved in")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="/home/data/ssh5131/focus_modeling/model/KL0_DPRBi_BART/epoch6-ppl8.9141.ckpt", help="Path of the model checkpoint")
    parser.add_argument("--retrieval_type", type=str, default="DPR", help="{DPR, TFIDF, TFIDF_sen, BM25, BM25_sen}")
    parser.add_argument("--DPR_train", action='store_true', help="DPR_train")
    parser.add_argument("--landmark_dic", type=str, default="/home/mnt/ssh5131/focus_modeling/our_data/all_landmark_dic.json", help="landmark_dic json file")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--max_length", type=int, default=40, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum length of the output utterances")
    parser.add_argument("--temperature", type=int, default=1.0, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--seed", type=int, default=19981014, help="Seed")
    parser.add_argument("--regen_question", type=bool, default=False)
    parser.add_argument("--factcc_model", type=str, default="/home/data/ssh5131/focus_modeling/factcc/factcc-checkpoint", help="pre-trained factcc model directory")
    parser.add_argument("--dae_model", type=str, default="/home/data/ssh5131/focus_modeling/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    args = vars(parser.parse_args())

    print(":: Fix Seed", args['seed'], " ::")
    seed_everything(args['seed'])
    print('args: ', args)

    from setproctitle import setproctitle
    setproctitle("suhyun eval")

    print("Get model and tokenizer")

    model = Model(**args)

    model.to('cuda')
    model.eval()


    trainer_args = {
        'num_sanity_val_steps': 2, #None if args['test_mode'] else 0
        'deterministic': False,
        'gpus': 1,
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

