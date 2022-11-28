#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from argparse import ArgumentParser
import wandb, json
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import Sigmoid, Softmax, CrossEntropyLoss
from baselines.FoCus.data_utils import get_leaderboard_testdata_loaders, add_special_tokens_test, special_tokens
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from sklearn.metrics import accuracy_score
from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
from baselines.FoCus.cusgen_generate import generate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix



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

        test_dataset = get_leaderboard_testdata_loaders(self.hparams, self.tokenizer)
        self.test_dataset = test_dataset

    def test_dataloader(self):
        print("\n::: Load and preprocess TEST dataset :::")
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset)
        test_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.hparams.test_batch_size)
        return test_loader


    def step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(**batch)
        result = {'lm_logits':output['dynamic_lm_logits'], 'knowledge_logits':output['knowledge_logits'], 'persona_logits':output['persona_logits']}

        return result

    def test_step(self, batch, batch_idx):
        input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, knowledge_candidates, \
        knowledge_can_idx, tot_knowledge, tot_knowledge_eos, dialog, ID = batch

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

        for batch_i in range(self.hparams.test_batch_size):
            only_dial_input_ids_batch = dialog[batch_i]
            mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
            only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
            if len(all_persona_pred[batch_i])>0:
                concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                new_persona = concat_persona
            else:
                new_persona = None

            new_knowledge = all_knowledge_pred[batch_i]

            if new_persona is not None:
                new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)
            else:
                new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch[1:], eos_tensor], dim=-1)

        with torch.no_grad():
            #out_ids = sample_sequence(new_input.unsqueeze(0), token_type_ids=None, decoder_input_ids=decoder_input_ids, tokenizer=self.tokenizer, model=self.model, args=self.hparams, current_output=None)
            num_beams = self.hparams.num_beams
            num_return_sequences = self.hparams.num_beams
            top_k = self.hparams.top_k
            out_ids = sample_sequence(new_input.unsqueeze(0), token_type_ids=None, decoder_input_ids=decoder_input_ids, tokenizer=self.tokenizer, model=self.model, args=self.hparams, current_output=None)
            #out_ids = generate(new_input.unsqueeze(0), tokenizer=self.tokenizer, model=self.congenmodel, num_beams=num_beams, num_return_sequences=num_return_sequences, top_k=top_k)

        model_pred_k = self.tokenizer.decode(new_knowledge.tolist(), skip_special_tokens=True)
        if num_beams > 1:
            out_ids = [self.tokenizer.decode(output_item, skip_special_tokens=True) for output_item in out_ids.tolist()]
        else:
            if type(out_ids) == list:
                out_ids = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            else:
                out_ids = self.tokenizer.decode(out_ids.squeeze(0).tolist(), skip_special_tokens=True)


        print('\n pred: ', out_ids, '\n model_k: ', model_pred_k)
        result = dict()
        result['pg_pred'] = p_index_cvtd
        result['kg_pred'] = k_index_1.squeeze(0)
        result['y_pred_text'] = out_ids
        result['model_pred_knowledge'] = model_pred_k
        result['id'] = ID
        # self.log('y_true', result['y_true'])
        return result


    def epoch_end(self, outputs, state='test'):
        text_result = []
        for index, i in enumerate(outputs):
            text_dict = dict()
            text_dict['pg_pred'] = i['pg_pred']
            text_dict['kg_pred'] = i['kg_pred']
            text_dict['y_pred_text'] = i['y_pred_text']
            text_dict['model_pred_knowledge'] = i['model_pred_knowledge']
            text_dict['id'] = i['id']

            text_result.append(text_dict)

        result = {'text_result':text_result}


        return result


    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')

        result_dict_gen = dict()
        result_dict_ground = dict()

        alllist_gen = list()
        alllist_ground = list()
        ID_set = set()

        text_result = result['text_result']

        for test_data_index, test_data in enumerate(text_result):
            outputdict_gen = dict()
            outputdict_ground = dict()
            p_index_cvtd = test_data['pg_pred']
            k_index_cvtd = test_data['kg_pred']
            pred_reply = test_data['y_pred_text']
            ID = test_data['id']
            ID = ID.tolist()[0]
            decoded_ID = [chr(x) for x in ID]
            decoded_ID = ''.join(decoded_ID)

            gen_dict = dict()
            gen_dict['generation'] = pred_reply
            pg_kg_dict = dict()
            pg_kg_dict['pg'] = [int(x) for x in p_index_cvtd.tolist()]
            pg_kg_dict['kg'] = k_index_cvtd.item()

            if decoded_ID not in ID_set:
                outputdict_gen[decoded_ID] = [gen_dict]
                outputdict_ground[decoded_ID] = [pg_kg_dict]
            else:
                for dict_item_gen in alllist_gen:
                    if decoded_ID in dict_item_gen.keys():
                        dict_item_gen[decoded_ID].append(gen_dict)
                for dict_item_ground in alllist_ground:
                    if decoded_ID in dict_item_ground.keys():
                        dict_item_ground[decoded_ID].append(pg_kg_dict)

            alllist_gen.append(outputdict_gen)
            alllist_ground.append(outputdict_ground)
            ID_set.add(decoded_ID)

        result_dict_gen['baseline'] = alllist_gen
        result_dict_ground['baseline'] = alllist_ground

        with open(self.hparams.output_dir + self.hparams.flag + '_generation.json', 'w') as genfile:
            json.dump(result_dict_gen, genfile, indent='\t')
        with open(self.hparams.output_dir + self.hparams.flag + '_grounding.json', 'w') as groundingfile:
            json.dump(result_dict_ground, groundingfile, indent='\t')
            print("done!")


        return result_dict_gen, result_dict_ground



def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="/home/mnt/yoonna/focus_modeling/data/test_focus_public.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='/home/mnt/yoonna/focus_modeling/data/test_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--output_dir", type=str, default="/home/mnt/yoonna/focus_modeling/official_output/", help="Path for the output file to be saved in")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--retrieval_type", type=str, default="TFIDF", help="{DPR, TFIDF, TFIDF_sen, BM25, BM25_sen}")
    parser.add_argument("--DPR_train", action='store_true', help="DPR_train")
    parser.add_argument("--landmark_dic", type=str, default="/home/mnt/ssh5131/FoCus_data/our_data/all_landmark_dic.json", help="landmark_dic json file")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--max_length", type=int, default=40, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum length of the output utterances")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--leaderboard", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--seed", type=int, default=19950604, help="Seed")
    args = vars(parser.parse_args())

    print(":: Fix Seed", args['seed'], " ::")
    seed_everything(args['seed'])
    print('args: ', args)

    from setproctitle import setproctitle
    setproctitle("Yoonna eval")

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

