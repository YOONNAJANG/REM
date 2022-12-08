from setproctitle import setproctitle
setproctitle("yoonna")
import sys


from argparse import ArgumentParser
import os
from itertools import chain
print(os.getcwd())
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import AdamW
from data_utils_refine import add_special_tokens_, special_tokens_focus, dataloader_focus, dataloader_wow
from collections import Counter, defaultdict
from ptuning import get_embedding_layer, PromptEncoder, get_vocab_by_strategy
from torch.nn import Softmax


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "ner_labels"]


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pseudo_token = self.hparams.pseudo_token

        from transformers import BartTokenizer, BartConfig
        from refiner_modules import BartEncDec
        self.config = BartConfig.from_pretrained(self.hparams.pretrained_model)
        self.model = BartEncDec.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.pretrained_model)
        # self.model.to(self.hparams.device)
        self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens=special_tokens_focus)
        #add_special_tokens_(self.model, self.tokenizer, special_tokens=)

        if self.hparams.ptuning==True:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                print('frozen params: ', name)
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

            # output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels, ner_labels=ner_labels)

        return output


    def embed_inputs(self, queries):
        bz = queries.shape[0] #batchsize
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #bsz, seqlen, embdim
        blocked_indices = (queries == self.pseudo_token_id)
        blocked_indices =  torch.nonzero(blocked_indices, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # True index tensors -> bz, spell_length, 2 -> :,:,1 (한 입력마다 해당 인덱스 불러옴) ->bsz, spell_length
        replace_embeds = self.prompt_encoder() #spell_length, embdim
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :] #해당 토큰 자리만 replace embedding
        return raw_embeds


    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids,
            'lm_labels':lm_labels,
            'ner_labels':ner_labels
        }
        result = self.step(inputs, batch_idx)
        lm_loss, ner_loss = result['lm_loss'], result['ner_loss']
        loss = (lm_loss * self.hparams.lm_coef + ner_loss * self.hparams.ner_coef) / self.hparams.grad_accum
        self.log('train_loss', loss)
        self.log('train_lm_loss', result['lm_loss'])
        self.log('train_ner_loss', result['ner_loss'])
        result['loss'] = loss
        result['lm_loss'] = lm_loss
        result['ner_loss'] = ner_loss
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        inputs = {
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
            # 'lm_labels': lm_labels,
            # 'ner_labels': ner_labels
        }
        result = self.step(inputs, batch_idx)

        #print(result.items())
        # result = {k: v.detach().cpu() for k, v in result.items()}

        lm_logits = result['lm_logits']
        ner_logits = result['ner_logits']
        softmax = Softmax(dim=-1)
        lm_pred = softmax(lm_logits)
        lm_val, lm_pred_idx = torch.topk(lm_pred, k=1, dim=-1)
        lm_pred_idx = lm_pred_idx.squeeze(-1)
        mask = (lm_labels != -100)
        lm_labels_only = [lm_labels[mask].tolist()]
        lm_pred_idx = lm_pred_idx[mask].tolist()
        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))

        ner_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        ner_loss = ner_criterion(ner_logits.view(-1, 6), ner_labels.view(-1).long())

        hst_index = (input_ids == 50266).nonzero(as_tuple=True)[1]

        ner_acc = 0
        for i in range(ner_logits.shape[0]):
            # print("hst_index[i]:  ",hst_index[i])
            logits_clean = ner_logits[i][:hst_index[i].item()]
            label_clean = ner_labels[i][:hst_index[i].item()]
            # print(logits_clean.size())
            # print(label_clean.size())

            predictions = logits_clean.argmax(dim=1)
            # print(predictions)
            # print(label_clean)
            acc = (predictions == label_clean).float().mean()
            ner_acc += acc

        self.log('valid_lm_loss', lm_loss)
        self.log('valid_ner_loss', ner_loss)

        result_dict = {
            'lm_loss':lm_loss.detach().cpu(),
            'ner_loss':ner_loss.detach().cpu(),
            'ner_acc':ner_acc.detach().cpu()
        }

        return result_dict

    def epoch_end(self, outputs, state='train'):
        if state=='train' or state=='val':
            lm_loss = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
            cls_loss = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
            ppl = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
            acc = torch.tensor(0, dtype=torch.float).to(self.hparams.device)

            for i in outputs:
                lm_loss += i['lm_loss']
                cls_loss += i['ner_loss']
                ppl += torch.exp(i['lm_loss'])
                acc += i['ner_acc']

            lm_loss = lm_loss / len(outputs)
            ner_loss = cls_loss / len(outputs)
            ppl = ppl / len(outputs)
            acc = acc / len(outputs)


            result = {'lm_loss': lm_loss, 'ner_loss': ner_loss, 'ppl': ppl, 'acc': acc}

        return result

    def train_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='train')
        self.log('train_lm_loss', result['lm_loss'])
        self.log('train_ner_loss', result['ner_loss'])
        self.log('train_ppl', result['ppl'])
        self.log('train_acc', result['acc'])
        return result

    def validation_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='val')
        self.log('valid_lm_loss', result['lm_loss'])
        self.log('valid_ner_loss', result['ner_loss'])
        self.log('valid_ppl', result['ppl'])
        self.log('valid_acc', result['acc'])
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
            rain_dataset, valid_dataset = dataloader_wow(self.hparams, self.tokenizer)
        elif self.hparams.data_type == "persona":
            rain_dataset, valid_dataset = None, None
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
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, persona}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    parser.add_argument("--ckpt", type=str, default="facebook/bart-base", help="ckpt path") #facebook/bart-base
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=2)
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
    parser.add_argument("--template", type=str, default="3,3,3") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])

    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])

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
    wandb.init(project='focus_regen', reinit=True, config=args, settings=wandb.Settings(start_method='fork'))
    wandb_logger = WandbLogger(project='focus_regen')
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
        'precision': args['precision'],
        'logger': wandb_logger}

    if args['ckpt'] not in ['facebook/bart-base']:
        print(':: Load checkpoint from hparams ::')
        print(torch.load(args['ckpt'])['hyper_parameters'])
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['ckpt'])

    trainer = Trainer(**trainer_args)
    trainer.fit(model)
    # print(":: Start Testing ::")
    # with torch.no_grad():
    #     trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    main()