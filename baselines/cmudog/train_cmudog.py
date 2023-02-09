import os, math
import logging
from argparse import ArgumentParser
# import wandb
import torch
from torch.nn import Sigmoid, Softmax
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.optim import AdamW, Adam
from torch_optimizer import Adafactor
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils import get_data_loaders, add_special_tokens_, special_tokens
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


logger = logging.getLogger(__file__)

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # kwargs are saved to self.hparams

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer, BartForConditionalGeneration
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.model_path == "t5-base" or "t5-large":
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)


        elif self.hparams.model_name == 'transformer-encdec':
            from transformers import BartTokenizer, BartConfig
            from baselines.FoCus.cusgen_generate import BARTPK_ctxt as bartmodel
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model_config = BartConfig.from_pretrained(self.hparams.model_path)
            self.model = bartmodel(self.model_config)
            self.model.to(self.hparams.device)
            #if self.hparams.model_path == "facebook/bart-base" or "facebook/bart-large":
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)


        else:
            raise NotImplementedError

        train_dataset, valid_dataset = get_data_loaders(self.hparams, self.tokenizer)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        print("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))



    def train_dataloader(self):
        print("\n:: Load and preprocess TRAIN dataset ::")
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        train_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.hparams.train_batch_size)


        return train_loader

    def val_dataloader(self):
        print("\n:: Load and preprocess VALID dataset ::")
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset)
        valid_loader = DataLoader(self.valid_dataset, sampler=valid_sampler, batch_size=self.hparams.valid_batch_size)
        return valid_loader


    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'Adafactor':
            optimizer = Adafactor(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'lambdalr':
            scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
            #scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def step(self, batch, batch_idx):
        output = self.model(**batch)
        result = {
            'loss':output['loss'] if 'loss' in output else None,
            'logits':output['logits'] if 'logits' in output else None,
        }
        return result



    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels = batch

        # print("\ninput_ids.size():", input_ids.size())
        # print("\ninput_ids:", input_ids)
        # print("\ndecoder_input_ids.size():", decoder_input_ids.size())
        # print("\ndecoder_input_ids:", decoder_input_ids)
        # print("\nlm_labels.size():", lm_labels.size())
        # print("\nlm_labels:", lm_labels)
        # exit()
    
        inputs = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids, 'labels': lm_labels}
        result = self.step(inputs, batch_idx)
        loss = result['loss']
        logits = result['logits']
        loss = loss / self.hparams.grad_accum
        ppl = torch.exp(loss)

        self.log('train_loss', loss.item())
        self.log('train_ppl', ppl)

        result_dict = {
            'loss': loss,
            'logits': logits,
            'ppl': ppl
        }
        return result_dict


    def validation_step(self, batch, batch_idx):

        input_ids, decoder_input_ids, lm_labels = batch
        inputs = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids, 'labels': lm_labels}
        result = self.step(inputs, batch_idx)
        loss = result['loss']
        logits = result['logits']
        ppl = torch.exp(loss)

        self.log('train_loss', loss.item())
        self.log('train_ppl', ppl)


        result = {
            'loss':loss,
            'ppl':ppl
        }
        return result


    def epoch_end(self, outputs, state='train'):
        if state=='train':
            loss = torch.tensor(0, dtype=torch.float)
            for i in outputs:
                loss += i['loss'].cpu().detach()
            loss = loss / (len(outputs)+1)
            ppl = torch.exp(loss)
            result={
                'loss': loss,
                'ppl': ppl
            }
            self.log('train_loss', loss)
            self.log('train_ppl', ppl)
            print('\n\ntrain_loss:', loss)
            print('train_ppl:', ppl)

        elif state=='val':
            loss = torch.tensor(0, dtype=torch.float)
            for i in outputs:
                loss += i['loss'].cpu().detach()
            loss = loss / (len(outputs) + 1)
            ppl = torch.exp(loss)
            result = {
                'loss': loss,
                'ppl': ppl
            }
            self.log('valid_loss', loss)
            self.log('valid_ppl', ppl)
            print('\n\nvalid_loss:', loss)
            print('valid_ppl:', ppl)
        return result

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BART",
                        help="{BART, T5, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                    help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")  # or "facebook/bart-large"
    parser.add_argument("--checkpoint", type=str, default="",
                        help="load checkpoint and resume train")
    parser.add_argument("--train_dataset_path", type=str, default="data/train.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_path", type=str, default="data/valid_random_split.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--grad_accum", type=int, default=32,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="lambdalr", help="{exp, lambdalr}")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="{AdamW, AdamP}")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--flag", type=str, default="", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=19950604)
    parser.add_argument("--output_dir", type=str, default="/home/data/yoonna/focusmodeling/wow/model/", help="directory where the model to be saved on")
    args = vars(parser.parse_args())

    print(":: Using PyTorch Ver", torch.__version__, " ::")
    print(":: Fix Seed", args['seed'], " ::")
    from setproctitle import setproctitle
    setproctitle("leejeongwoo")

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

    if args['flag']:
        flag = args['flag']
    else:
        flag = 'E'+str(args['n_epochs'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['output_dir']+flag,
        filename='epoch{epoch}-ppl{valid_ppl:.4f}',
        monitor='valid_ppl',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor='valid_ppl',
        patience=2,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor()


    # wandb.init(project='focus_modeling', reinit=True, config=args)
    # wandb_logger = WandbLogger(project='focus_modeling')
    # wandb.watch(model, log_freq=10)


    trainer_args = {
        'callbacks': [checkpoint_callback, early_stopping, lr_monitor],
        'max_epochs': args['n_epochs'],
        'fast_dev_run': args['test_mode'],
        'num_sanity_val_steps': 2, #None if args['test_mode'] else 0
        'accumulate_grad_batches': args['grad_accum'],
        'gradient_clip_val': args['max_norm'],
        'deterministic': False,
        'gpus': args['gpu_num'],
        'strategy': DDPPlugin(find_unused_parameters=True),
        'precision': 32}#,
        # 'logger': wandb_logger}


    if args['checkpoint']:
        print(':: Load checkpoint from hparams :')
        print(torch.load(args['checkpoint'])['hyper_parameters'])
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['checkpoint'])

    print(":: Start Training ::")
    trainer = Trainer(**trainer_args)

    trainer.fit(model)
    # wandb.finish()


if __name__ == "__main__":
    main()
