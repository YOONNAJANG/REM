from setproctitle import setproctitle
setproctitle("yoonna")
import sys

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
from argparse import ArgumentParser
from itertools import chain
print(os.getcwd())
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from data_utils_refine_yoonna import add_special_tokens_, special_tokens_focus, dataloader_focus
from collections import Counter, defaultdict
from ptuning import get_embedding_layer, PromptEncoder, get_vocab_by_strategy
from baselines.FoCus.data_utils import get_testdata_loaders


MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "ner_labels"]


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pseudo_token = self.hparams.pseudo_token

        from transformers import BartTokenizer, BartConfig
        from refiner_modules_yoonna import BartEncDec
        self.config = BartConfig.from_pretrained(self.hparams.pretrained_model)
        self.model = BartEncDec.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.pretrained_model)
        # self.model.to(self.hparams.device)
        self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens=special_tokens_focus)
        #add_special_tokens_(self.model, self.tokenizer, special_tokens=)
        print('hparams: ', self.hparams)
        print('ptuning: ', self.hparams.ptuning)
        if self.hparams.ptuning==True:
            for name, param in self.model.named_parameters():
                # print('not frozen params: ', name)
                # if name.startswith('model.encoder.'):
                param.requires_grad = False
            self.embeddings = get_embedding_layer(self.hparams, self.model)
            # set allowed vocab set
            self.vocab = self.tokenizer.get_vocab()
            self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.hparams, self.tokenizer))
            self.template = tuple([int(item) for item in self.hparams.template.split(',')])
            # load prompt encoder
            self.hidden_size = self.embeddings.embedding_dim
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pseudo_token]})

            self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            self.spell_length = sum(self.template)
            self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.hparams.device, self.hparams)
            self.prompt_encoder = self.prompt_encoder.to(self.hparams.device)

        if len(self.hparams.checkpoint) > 0:
            checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
            # breakpoint()
            self.checkpoint_loaded = dict()
            self.checkpoint_prompt = dict()
            for k, v in checkpoint.items():
                if k.startswith('model.'):
                    self.checkpoint_loaded[k[6:]] = v
                else:
                    self.checkpoint_prompt[k] = v

            self.model.load_state_dict(self.checkpoint_loaded)
            # self.congenmodel.load_state_dict(checkpoint, strict=False)
        test_dataset = get_testdata_loaders(self.hparams, self.tokenizer)
        self.test_dataset = test_dataset

    def embed_inputs(self, queries):
        bz = queries.shape[0] #batchsize
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #bsz, seqlen, embdim
        blocked_indices = (queries == self.pseudo_token_id)
        blocked_indices = torch.nonzero(blocked_indices, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # True index tensors -> bz, spell_length, 2 -> :,:,1 (한 입력마다 해당 인덱스 불러옴) ->bsz, spell_length
        replace_embeds = self.prompt_encoder() #spell_length, embdim
        self.prompt_encoder.load_state_dict(self.checkpoint_prompt)
        breakpoint()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :] #해당 토큰 자리만 replace embedding
        return raw_embeds


    def step(self, batch, batch_idx):
        # input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(batch['input_ids'])
            if 'input_ids' in batch:
                del batch['input_ids']
                batch['inputs_embeds'] = input_embeds
            output = self.model(**batch)
        else:
            output = self.model(**batch)
        return output

    def generation_step(self, batch, batch_idx):
        # input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(batch['input_ids'])
            if 'input_ids' in batch:
                del batch['input_ids']
                batch['inputs_embeds'] = input_embeds
                breakpoint()
            output = self.model(**batch)
        else:
            output = self.model(**batch)
        return output

    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        mask = (lm_labels != self.tokenizer.pad_token_id)
        reply = lm_labels[mask]
        inputs = {
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
            # 'lm_labels': lm_labels,
            # 'ner_labels': ner_labels
        }
        with torch.no_grad():
            results = self.step(inputs, batch_idx)
            generated = self.generation_step(inputs, batch_idx)

        lm_logits, ner_logits = results['lm_logits'], results['ner_logits']

        lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(lm_loss)

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
            "accuracy": results["overall_accuracy"]}

        result = {}
        for k, v in results.items():
            if k != "ner_results":
                result[k] = v.detach().cpu()
            else:
                result[k] = v

        self.log('test_ppl', result['ppl'])
        self.log('true_text', result['reply'])
        self.log('pred_text', result["ner_results"])
        self.log('input_text', result["ner_results"]["f1"])
        self.log('test_ner_accuracy', result["ner_results"]["accuracy"])
        self.log('test_ner_f1', result["ner_results"]["f1"])
        self.log('test_ner_recall', result["ner_results"]["recall"])
        self.log('test_ner_precision', result["ner_results"]["precision"])
        wandb.log({'test_ppl': result['ppl']})
        wandb.log({'true_text': result['reply']})
        wandb.log({'pred_text': result['ner_results']})
        wandb.log({'true_text': result['reply']})
        wandb.log({'test_ner_accuracy': result["ner_results"]["accuracy"]})
        wandb.log({'test_ner_f1': result["ner_results"]["f1"]})
        wandb.log({'test_ner_recall': result["ner_results"]["recall"]})
        wandb.log({'test_ner_precision': result["ner_results"]["precision"]})

        return result

    def epoch_end(self, outputs, state='train'):
        lm_loss = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
        cls_loss = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
        ppl = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
        ner_acc = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
        ner_f1 = torch.tensor(0, dtype=torch.float).to(self.hparams.device)


        for i in outputs:
            lm_loss += i['lm_loss']
            cls_loss += i['ner_loss']
            ppl += torch.exp(i['lm_loss'])
            ner_acc += i["ner_results"]["accuracy"]
            ner_f1 += i["ner_results"]["f1"]

        lm_loss = lm_loss / len(outputs)
        ner_loss = cls_loss / len(outputs)
        ppl = ppl / len(outputs)
        ner_acc = ner_acc / len(outputs)
        ner_f1 = ner_f1 / len(outputs)


        result = {'lm_loss': lm_loss, 'ner_loss': ner_loss, 'ppl': ppl, 'ner_acc': ner_acc, 'ner_f1': ner_f1}

        return result

    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')
        self.log('valid_lm_loss', result['lm_loss'])
        self.log('valid_ner_loss', result['ner_loss'])
        self.log('valid_ppl', result['ppl'])
        self.log('valid_ner_acc', result['ner_acc'])
        self.log('valid_ner_f1', result['ner_f1'])
        wandb.log({'valid_lm_loss': result['lm_loss']})
        wandb.log({'valid_ner_loss': result['ner_loss']})
        wandb.log({'valid_ppl': result['ppl']})
        wandb.log({'valid_ner_acc': result['ner_acc']})
        wandb.log({'valid_ner_f1': result['ner_f1']})
        return result

    def dataloader(self):
        if self.hparams.data_type == "focus":
            train_dataset, valid_dataset = dataloader_focus(self.hparams, self.tokenizer)
        elif self.hparams.data_type == "wow":
            rain_dataset, valid_dataset = None, None
        elif self.hparams.data_type == "persona":
            rain_dataset, valid_dataset = None, None
        return train_dataset, valid_dataset

    def test_dataloader(self):
        print("\n::: Load and preprocess TEST dataset :::")
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        test_loader = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.hparams.test_batch_size)
        return test_loader

def main():

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, persona}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    parser.add_argument("--checkpoint", type=str, default="", help="ckpt path") #facebook/bart-base
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--random_seed", type=int, default=644128)
    parser.add_argument("--lm_coef", type=float, default=1.0, help="Coefficient for LM loss")
    parser.add_argument("--ner_coef", type=float, default=1.0, help="Coefficient for NER loss")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="{AdamW, AdamP}")
    parser.add_argument("--lr_scheduler", type=str, default="lambdalr", help="{exp, lambdalr}")
    parser.add_argument("--grad_accum", type=int, default=32, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--precision", type=int, default=32, help="{16,32,64}")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="", help="default value for PLMs")


    #for p-tuning
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="50,50,50") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])




    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])

    model = Model(**args)
    model.eval()
    model.to(args['device'])


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