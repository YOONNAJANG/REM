import os
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

import sys
sys.path.append('/home/data/yoonna/Refiner')
from metrics.dae_factuality.evaluate_factuality import score_example_single_context
from metrics.distinctN import distinct_n_sentence_level


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer
            from baselines.FoCus.cusgen_generate import Bart_pkgen as bartmodel
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
            from baselines.FoCus.cusgen_generate import T5_pkgen as t5model
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
            from baselines.FoCus.cusgen_generate import LED_pkgen as ledmodel
            self.tokenizer = LEDTokenizer.from_pretrained(self.hparams.model_path)
            self.model = ledmodel.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'transformer-encdec':
            from transformers import BartTokenizer, BartConfig
            from baselines.FoCus.cusgen_generate import BARTPK_ctxt as bartmodel
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
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        test_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.hparams.test_batch_size)
        return test_loader


    def step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(**batch)
        result = {'lm_logits':output['dynamic_lm_logits'], 'knowledge_logits':output['knowledge_logits'], 'persona_logits':output['persona_logits']}

        return result

    def test_step(self, batch, batch_idx):
        input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
        knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog, history_list = batch

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

        persona, knowledge = self.tokenizer.convert_tokens_to_ids(list(special_tokens.values())[-2:])
        bos, padding, eos = 0, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id  # 0, 1, 2

        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        bos_tensor = torch.tensor([bos]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)
        max_position = 1024

        softmax = Softmax(dim=-1)
        knowledge_softmax = softmax(knowledge_logits)
        _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
        all_knowledge_pred = []

        for batch_i in range(self.hparams.test_batch_size):
            knowledge_pred_idx = k_index_1[batch_i]
            knowledge_can_txt = self.tokenizer.batch_decode(knowledge_candidates.cpu().tolist()[0], skip_special_tokens=True)

        mask_input = torch.ne(input_ids, padding)
        input_wo_mask = torch.masked_select(input_ids, mask_input)
        input_ids_txt = self.tokenizer.decode(input_wo_mask, skip_special_tokens=False)


        result = dict()


        result['kg_pred'] = k_index_1.squeeze(0)
        result['kg_true'] = knowledge_grounding
        result['input_text'] = input_ids_txt
        result['candidate'] = knowledge_can_txt


        return result


    def epoch_end(self, outputs, state='test'):
        text_result = []

        for index, i in enumerate(outputs):
            text_dict = dict()
            text_dict['kg_pred'] = i['kg_pred']
            text_dict['kg_true'] = i['kg_true']
            text_dict['input_text'] = i['input_text']
            text_dict['candidate'] = i['candidate']

            text_result.append(text_dict)

        result = {'text_result': text_result}


        return result


    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')

        text_result = result['text_result']
        kg_score = 0

        result_list = list()

        for test_data_index, test_data in enumerate(text_result):
            pred_dict = dict()
            k_index_cvtd = test_data['kg_pred']
            knowledge_grounding = test_data['kg_true']
            input = test_data['input_text']
            candidate = test_data['candidate']
            pred_dict['input'] = input
            pred_dict['candidate'] = candidate
            pred_dict['true'] = knowledge_grounding.item()
            pred_dict['pred'] = k_index_cvtd.item()
            result_list.append(pred_dict)

            # KG
            kg_acc = accuracy_score(knowledge_grounding.cpu(), k_index_cvtd.cpu())
            kg_score += kg_acc
            print(test_data_index, '\tinput: ', input, '\tcandidate: ', candidate, '\tpred: ', k_index_cvtd, '\ttrue: ', knowledge_grounding, '\tT/F: ', kg_acc)

        kg_result = kg_score/(test_data_index+1)
        print('KG hit@1: ', kg_result)

        result_dict = dict()
        result_dict['kg_acc'] = kg_result
        result_dict['output'] = result_list

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')

        return test_result



def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="/home/mnt/ssh5131/FoCus_data/our_data/test_ours.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='/home/mnt/ssh5131/FoCus_data/our_data/focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--output_dir", type=str, default="/home/mnt/ssh5131/FoCus_modeling/eval_output/", help="Path for the output file to be saved in")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--retrieval_type", type=str, default="TFIDF", help="{DPR, TFIDF, TFIDF_sen, BM25, BM25_sen}")
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
    parser.add_argument("--seed", type=int, default=19950604, help="Seed")
    parser.add_argument("--regen_question", type=bool, default=False)
    parser.add_argument("--factcc_model", type=str, default="/home/mnt/yoonna/focus_modeling/factcc/factcc-checkpoint", help="pre-trained factcc model directory")
    parser.add_argument("--dae_model", type=str, default="/home/mnt/yoonna/focus_modeling/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    args = vars(parser.parse_args())

    print(":: Fix Seed", args['seed'], " ::")
    seed_everything(args['seed'])
    print('args: ', args)

    from setproctitle import setproctitle
    setproctitle("yoonna eval")

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