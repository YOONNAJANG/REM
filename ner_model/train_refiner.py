from setproctitle import setproctitle
setproctitle("suhyun")
import sys

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

from argparse import ArgumentParser
import os
from itertools import chain
print(os.getcwd())
import wandb
import torch
torch.set_num_threads(4)
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import AdamW
from collections import Counter, defaultdict
import numpy as np
import random
from ptuning import get_embedding_layer, PromptEncoder, get_vocab_by_strategy, init_prompt_embedding, init_focus_tokens_embedding
from data_utils_refine import add_special_tokens_, special_tokens_focus, dataloader_focus, dataloader_wow
# dataloader_cmudog


MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "ner_labels"]


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pseudo_token = self.hparams.pseudo_token

        from transformers import AutoTokenizer, BartConfig
        if self.hparams.mode == "ner":
            from refiner_modules import BartEncDec as bartmodel
        elif self.hparams.mode == "gen_exp":
            from refiner_modules import BartEncDec_NER_explicit as bartmodel
        elif self.hparams.mode == "gen_imp":
            from refiner_modules import BartEncDec_NER_implicit as bartmodel
        else:
            raise NotImplementedError

        self.config = BartConfig.from_pretrained(self.hparams.pretrained_model)
        self.model = bartmodel.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        # self.model.to(self.hparams.device)
        self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens=special_tokens_focus)
        #add_special_tokens_(self.model, self.tokenizer, special_tokens=)
        print('hparams: ', self.hparams)
        print('ptuning: ', self.hparams.ptuning)
        print('mode: ', self.hparams.mode)
        if self.hparams.ptuning==True:
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer,
                                                             special_tokens={'pseudo_token':self.pseudo_token})

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
            self.model.resize_token_embeddings(len(self.tokenizer))

            self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
            self.spell_length = sum(self.template)
            self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.hparams.device, self.hparams)
            self.prompt_encoder = self.prompt_encoder.to(self.hparams.device)

            if len(self.hparams.target_word_to_init) > 0:
                init_prompt_embedding(self.pseudo_token_id, self.hparams.target_word_to_init, self.model, self.tokenizer)

        init_focus_tokens_embedding(special_tokens_focus, self.model, self.tokenizer)


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

            self.model.load_state_dict(self.checkpoint_loaded, strict=False)

    def embed_inputs(self, queries):
        bz = queries.shape[0] #batchsize
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #bsz, seqlen, embdim
        blocked_indices = (queries == self.pseudo_token_id)
        blocked_indices = torch.nonzero(blocked_indices, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # True index tensors -> bz, spell_length, 2 -> :,:,1 (한 입력마다 해당 인덱스 불러옴) ->bsz, spell_length
        replace_embeds = self.prompt_encoder() #spell_length, embdim
        if len(self.hparams.checkpoint) > 0:
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


    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids,
            'labels':lm_labels,
            'ner_labels':ner_labels
        }
        result = self.step(inputs, batch_idx)
        lm_loss, ner_loss = result['lm_loss'], result['ner_loss']
        loss = (lm_loss * self.hparams.lm_coef + ner_loss * self.hparams.ner_coef) / self.hparams.grad_accum
        self.log('train_loss', loss)
        self.log('train_lm_loss', result['lm_loss'])
        self.log('train_ner_loss', result['ner_loss'])
        self.log('train_ner_acc', result["ner_results"]["accuracy"])
        self.log('train_ner_f1', result["ner_results"]["f1"])
        self.log('train_ner_recall', result["ner_results"]["recall"])
        self.log('train_ner_precision', result["ner_results"]["precision"])

        wandb.log({'train_loss': loss})
        wandb.log({'train_lm_loss': result['lm_loss']})
        wandb.log({'train_ner_loss': result['ner_loss']})
        wandb.log({'train_ner_acc': result["ner_results"]["accuracy"]})
        wandb.log({'train_ner_f1': result["ner_results"]["f1"]})
        wandb.log({'train_ner_recall': result["ner_results"]["recall"]})
        wandb.log({'train_ner_precision': result["ner_results"]["precision"]})

        result['loss'] = loss
        result['lm_loss'] = lm_loss
        result['ner_loss'] = ner_loss
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids,
            'labels':lm_labels,
            'ner_labels':ner_labels
        }
        results = self.step(inputs, batch_idx)
        #print(result.items())

        result = {}
        for k, v in results.items():
            if k != "ner_results":
                result[k] = v.detach().cpu()
            else:
                result[k] = v
        self.log('valid_lm_loss', result['lm_loss'])
        self.log('valid_ner_loss', result['ner_loss'])
        self.log('valid_ner_acc', result["ner_results"]["accuracy"])
        self.log('valid_ner_f1', result["ner_results"]["f1"])
        self.log('valid_ner_recall', result["ner_results"]["recall"])
        self.log('valid_ner_precision', result["ner_results"]["precision"])
        wandb.log({'valid_lm_loss': result['lm_loss']})
        wandb.log({'valid_ner_loss': result['ner_loss']})
        wandb.log({'valid_ner_acc': result["ner_results"]["accuracy"]})
        wandb.log({'valid_ner_f1': result["ner_results"]["f1"]})
        wandb.log({'valid_ner_recall': result["ner_results"]["recall"]})
        wandb.log({'valid_ner_precision': result["ner_results"]["precision"]})

        return result

    def epoch_end(self, outputs, state='train'):

        if state=='train' or state=='val':
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

    def train_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='train')
        self.log('train_lm_loss', result['lm_loss'])
        self.log('train_ner_loss', result['ner_loss'])
        self.log('train_ppl', result['ppl'])
        self.log('train_ner_acc', result['ner_acc'])
        self.log('train_ner_f1', result['ner_f1'])
        wandb.log({'train_lm_loss': result['lm_loss']})
        wandb.log({'train_ner_loss': result['ner_loss']})
        wandb.log({'train_ppl': result['ppl']})
        wandb.log({'train_ner_acc': result['ner_acc']})
        wandb.log({'train_ner_f1': result['ner_f1']})
        return result

    def validation_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='val')
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

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'lambdalr':
            scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def dataloader(self):

        if self.hparams.data_type == "focus":
            train_dataset, valid_dataset = dataloader_focus(self.hparams, self.tokenizer)
        elif self.hparams.data_type == "wow":
            train_dataset, valid_dataset = dataloader_wow(self.hparams, self.tokenizer)
        elif self.hparams.data_type == "cmudog":
            train_dataset, valid_dataset = dataloader_cmudog(self.hparams, self.tokenizer)
        return train_dataset, valid_dataset

    def train_dataloader(self):
        train_dataset, _ = self.dataloader()
        print("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        _, valid_dataset = self.dataloader()
        print("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))
        return DataLoader(valid_dataset, batch_size=self.hparams.valid_batch_size, shuffle=False)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, cmudog}")
    parser.add_argument("--mode", type=str, default="ner", help="{ner, gen_exp, gen_imp}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint path")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=4)
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
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/regen_add_ner/", help="default value for PLMs")

    #for p-tuning
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="50,50,50") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])
    parser.add_argument("--target_word_to_init", type=str, default="")

    #few shot setting
    parser.add_argument("--fewshot", type=bool, default=False)
    parser.add_argument("--fewnum", type=int, default=100)



    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])

    seed_everything(args['random_seed'])

    model = Model(**args)
    model.eval()
    model.to(args['device'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['output_dir'],
        filename='epoch{epoch}-valid_lm_loss{valid_lm_loss:.4f}',
        monitor='valid_lm_loss',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor='valid_lm_loss',
        patience=2,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor()


    print(":: Start Training ::")
    wandb.init(project='Refiner_ner', reinit=True, config=args, settings=wandb.Settings(start_method='fork'))
    # wandb.init(project='Refiner_ner', reinit=True, config=args)
    wandb_logger = WandbLogger(project='Refiner')
    wandb.watch(model, log_freq=20)

    trainer_args = {
        'callbacks': [checkpoint_callback, early_stopping, lr_monitor],
        'max_epochs': args['epochs'],
        'fast_dev_run': args['test_mode'],
        'num_sanity_val_steps': -1, #2: 2steps, -1: check all val data, 0: turn it off
        'accumulate_grad_batches': args['grad_accum'],
        'gradient_clip_val': args['max_norm'],
        'deterministic': torch.cuda.is_available(),
        'gpus': args['gpu_num'],
        'strategy': DDPPlugin(find_unused_parameters=True),
        'precision': args['precision'],}
        # 'logger': wandb_logger}


    trainer = Trainer(**trainer_args)
    trainer.fit(model)
    # print(":: Start Testing ::")
    # with torch.no_grad():
    #     trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    main()