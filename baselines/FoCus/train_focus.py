#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import os, math
import logging
from argparse import ArgumentParser
import wandb
import torch
from torch.nn import Sigmoid, Softmax
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.optim import AdamW, Adam
from torch_optimizer import Adafactor
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from baselines.FoCus.data_utils import get_data_loaders, add_special_tokens_, special_tokens
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


logger = logging.getLogger(__file__)

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # kwargs are saved to self.hparams

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer
            from baselines.FoCus.cusgen_generate import Bart_pkgen as bartmodel
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = bartmodel.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer
            from baselines.FoCus.cusgen_generate import T5_pkgen as t5model
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = t5model.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.model_path == "t5-base" or "t5-large":
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'LED':
            from transformers import LEDTokenizer
            from baselines.FoCus.cusgen_generate import LED_pkgen as ledmodel
            self.tokenizer = LEDTokenizer.from_pretrained(self.hparams.model_path)
            self.model = ledmodel.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.model_path == "allenai/led-base-16384" or "allenai/led-large-16384":
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


        train_dataset, valid_dataset = get_data_loaders(self.hparams, self.tokenizer, get_aug_data=self.hparams.get_aug_data)
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


    def forward(self, **kwargs):
        result = self.model(**kwargs)
        return result

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
            'lm_loss':output['lm_loss'] if 'lm_loss' in output else None,
            'kg_loss':output['knowledge_loss'] if 'knowledge_loss' in output else None,
            'pg_loss':output['persona_loss'] if 'persona_loss' in output else None,
            'kldiv_loss': output['kldiv_loss'] if 'kldiv_loss' in output else None,
            'lm_logits':output['lm_logits'] if 'lm_logits' in output else None,
            'dynamic_lm_logits':output['dynamic_lm_logits'] if 'dynamic_lm_logits' in output else None,
            'kg_logits':output['knowledge_logits'] if 'knowledge_logits' in output else None,
            'pg_logits':output['persona_logits'] if 'persona_logits' in output else None,
            'lm_labels':output['lm_labels'] if 'lm_labels' in output else None
        }
        return result



    def training_step(self, batch, batch_idx):

        input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
        knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog, history_list = batch
        inputs = {
            'input_ids':input_ids,
            'input_eos':input_eos,
            'only_dial_input_ids':dialog,
            'decoder_input_ids':decoder_input_ids,
            'persona_input_ids':persona_candidates,
            'knowledge_input_ids':knowledge_candidates,
            'persona_can_idx':persona_can_idx,
            'persona_grounding':persona_grounding,
            'knowledge_can_idx':knowledge_can_idx,
            'knowledge_grounding':knowledge_grounding,
            'tot_knowledge':tot_knowledge,
            'tot_knowledge_eos':tot_knowledge_eos,
            'lm_labels':lm_labels,
            'training':True
        }

        result = self.step(inputs, batch_idx)
        lm_loss, knowledge_loss, persona_loss, kldiv_loss = result['lm_loss'], result['kg_loss'], result['pg_loss'], result['kldiv_loss']
        loss = (lm_loss * self.hparams.lm_coef + knowledge_loss * self.hparams.kn_coef + persona_loss * self.hparams.ps_coef + kldiv_loss * self.hparams.kl_coef) / self.hparams.grad_accum
        ppl = torch.exp(lm_loss)

        self.log('train_loss', loss.item())
        self.log('train_ppl', ppl)
        self.log('train_lm_loss', lm_loss.item())
        self.log('train_kg_loss', knowledge_loss.item())
        self.log('train_pg_loss', persona_loss.item())
        self.log('train_kldiv_loss', kldiv_loss.item())

        result_dict = {
            'loss': loss
            # 'lm_loss':lm_loss,
            # 'kg_loss':knowledge_loss,
            # 'pg_loss':persona_loss
        }
        return result_dict


    def validation_step(self, batch, batch_idx):

        input_ids, input_eos, decoder_input_ids, lm_labels, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
        knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog, history_list = batch
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
            'training':False
        }
        result = self.step(inputs, batch_idx)
        lm_logits, knowledge_logits, persona_logits = result['dynamic_lm_logits'], result['kg_logits'], result['pg_logits']

        #print('lm logits: ', lm_logits, 'kg logits: ', knowledge_logits, 'pg logits: ', persona_logits)
        lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)

        persona_logits = persona_logits.squeeze()
        persona_grounding = persona_grounding.type_as(persona_logits).squeeze()

        sigmoid = Sigmoid()
        persona_pred_sigmoid = sigmoid(persona_logits)
        persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()

        softmax = Softmax(dim=-1)
        knowledge_pred = softmax(knowledge_logits)
        _, k_index_1 = torch.topk(knowledge_pred, k=1, dim=-1)
        #print('k index 1: ', k_index_1.size())
        k_index_1 = k_index_1.squeeze(-1)


        lm_pred = softmax(lm_logits_flat_shifted)
        lm_val, lm_idx = torch.topk(lm_pred, k=1, dim=-1)
        lm_idx = lm_idx.squeeze(-1)

        mask = (lm_labels_flat_shifted != -100)
        lm_labels_only = [lm_labels_flat_shifted[mask].tolist()]
        lm_idx_only = lm_idx[mask].tolist()

        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        kg_criterion = torch.nn.CrossEntropyLoss()
        kg_loss = kg_criterion(knowledge_logits, knowledge_grounding)
        pg_criterion = torch.nn.BCEWithLogitsLoss()
        pg_loss = pg_criterion(persona_logits, persona_grounding.type_as(persona_logits))

        # print("\n\npersona_grounding:", persona_grounding)
        # print("persona_grounding.type_as(persona_logits):", persona_grounding.type_as(persona_logits))
        # print("persona_grounding.view(-1):", persona_grounding.view(-1))
        # print("persona_grounding:", persona_grounding)
        # print("persona_pred_sigmoid:", persona_pred_sigmoid)
        pg_pred = persona_pred_sigmoid.view(-1)
        pg_true = persona_grounding.view(-1)


        #print('knowledge grounding: ', knowledge_grounding, 'knowledge pred: ', k_index_1)
        kg_acc = torch.sum(knowledge_grounding==k_index_1).item() / (len(knowledge_grounding) * 1.0)
        pg_acc = torch.sum(persona_grounding.type_as(persona_logits)==persona_pred_sigmoid).item() / (len(persona_grounding.view(-1)) * 1.0)
        ppl = torch.exp(lm_loss)

        result = {
            'lm_loss':lm_loss,
            'kg_loss': kg_loss,
            'pg_loss': pg_loss,
            'kg_acc':kg_acc,
            'pg_acc':pg_acc,
            'ppl':ppl,
            'pred':lm_idx_only,
            'target':lm_labels_only,
            'pg_true': pg_true,
            'pg_pred': pg_pred
        }
        return result


    def epoch_end(self, outputs, state='train'):
        if state=='train':
            loss = torch.tensor(0, dtype=torch.float)
            lm_loss = torch.tensor(0, dtype=torch.float)
            kg_loss = torch.tensor(0, dtype=torch.float)
            pg_loss = torch.tensor(0, dtype=torch.float)
            for i in outputs:
                lm_loss += i['lm_loss'].cpu().detach()
                kg_loss += i['kg_loss'].cpu().detach()
                pg_loss += i['pg_loss'].cpu().detach()
                #loss += i['loss'].cpu().detach()
            lm_loss = lm_loss / len(outputs)
            kg_loss = kg_loss / len(outputs)
            pg_loss = pg_loss / len(outputs)
            ppl = torch.exp(lm_loss)
            result={
                'lm_loss': lm_loss,
                'kg_loss': kg_loss,
                'pg_loss': pg_loss,
                'ppl': ppl
            }

        elif state=='val':
            lm_loss = torch.tensor(0, dtype=torch.float)
            kg_loss = torch.tensor(0, dtype=torch.float)
            pg_loss = torch.tensor(0, dtype=torch.float)
            kg_acc = torch.tensor(0, dtype=torch.float)
            pg_acc = torch.tensor(0, dtype=torch.float)
            bleu = torch.tensor(0, dtype=torch.float)
            rouge = torch.tensor(0, dtype=torch.float)
            f1 = torch.tensor(0, dtype=torch.float)

            pg_true_list = []
            pg_pred_list = []
            for i in outputs:
                pg_true_list.extend(i['pg_true'].tolist())
                pg_pred_list.extend(i['pg_pred'].tolist())

                lm_loss += i['lm_loss'].cpu().detach()
                kg_loss += i['kg_loss'].cpu().detach()
                pg_loss += i['pg_loss'].cpu().detach()
                kg_acc += i['kg_acc']
                #print('kg acc type: ', type(i['kg_acc']))
                pg_acc += i['pg_acc']
                pred_ = i['pred']
                target_ = i['target']
                #print('pred_', pred_)
            lm_loss = lm_loss / len(outputs)
            ppl = torch.exp(lm_loss)
            kg_acc = kg_acc / len(outputs)
            pg_acc = pg_acc / len(outputs)

            confusion = confusion_matrix(pg_true_list, pg_pred_list)
            accuracy = accuracy_score(pg_true_list, pg_pred_list)
            precision = precision_score(pg_true_list, pg_pred_list)
            recall = recall_score(pg_true_list, pg_pred_list)
            f1 = f1_score(pg_true_list, pg_pred_list)

            self.log('valid_lm_loss', lm_loss)
            self.log('valid_ppl', ppl)
            self.log('valid_kg_loss', kg_loss)
            self.log('valid_pg_loss', pg_loss)
            self.log('valid_kg_acc', kg_acc)
            self.log('valid_pg_acc', pg_acc)
            self.log('valid_pg_f1', f1)



            print('\n\nvalid_lm_loss:', lm_loss)
            print('valid_ppl:', ppl)
            print('valid_kg_loss:', kg_loss)
            print('valid_pg_loss:', pg_loss)
            print('valid_kg_acc:', kg_acc)
            print('valid_pg_acc:', pg_acc)

            #print("\npg_true_list:", pg_true_list)
            #print("pg_pred_list:", pg_pred_list)

            #print('valid_pg_Confusion_Matrix:')
            #print(confusion)
            print('valid_pg_accuracy:', accuracy)
            print('valid_pg_precision:', precision)
            print('valid_pg_recall:', recall)
            print('valid_pg_f1:', f1)



            result={
                'ppl': ppl,
                'kg_acc': kg_acc,
                'pg_acc': pg_acc
            }
            # self.log(state + '_loss', float(loss), on_epoch=True, prog_bar=True)
            # self.log(state + '_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
            # self.log(state + '_precision', precision_score(y_true, y_pred), on_epoch=True, prog_bar=True)
            # self.log(state + '_recall', recall_score(y_true, y_pred), on_epoch=True, prog_bar=True)
            # self.log(state + '_f1', f1_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        return result

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="",
                        help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                    help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")  # or "facebook/bart-large"
    parser.add_argument("--checkpoint", type=str, default="",
                        help="load checkpoint and resume train")
    parser.add_argument("--train_dataset_path", type=str, default="train_ours.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--train_dataset_cache", type=str, default='focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--dev_dataset_path", type=str, default="valid_ours.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_cache", type=str, default='focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--landmark_dic", type=str, default="./retrieval/all_landmark_dic.json", help="landmark_dic json file")
    parser.add_argument("--retrieval_type", type=str, default="DPR",
                        help="{DPR, TFIDF, TFIDF_sen, BM25, BM25_sen}")
    parser.add_argument("--use_knowledge_embedidngs", action='store_true', help="use precomputed knowledge embedding")
    parser.add_argument("--DPR_train", action='store_true', help="DPR_train")
    parser.add_argument("--DPR_model_path", type=str, default="./retrieval/sample_model/epoch_1.pt",
                    help="trained model path")
    parser.add_argument("--ps_coef", type=float, default=1.0, help="Coefficient for persona loss")
    parser.add_argument("--kl_coef", type=float, default=0.0, help="Coefficient for kldiv loss")
    parser.add_argument("--kn_coef", type=float, default=1.0, help="Coefficient for knowledge loss")
    parser.add_argument("--lm_coef", type=float, default=10.0, help="Coefficient for LM loss")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--grad_accum", type=int, default=16,
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
    parser.add_argument("--random_knowledge", action='store_true',
                        help="If true, the model choose the knowledge randomly")
    parser.add_argument("--get_aug_data", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="/home/mnt/yoonna/focus_modeling/model/", help="directory where the model to be saved on")
    args = vars(parser.parse_args())

    print(":: Using PyTorch Ver", torch.__version__, " ::")
    print(":: Fix Seed", args['seed'], " ::")
    from setproctitle import setproctitle
    setproctitle("yoonna")

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


    wandb.init(project='focus_modeling', reinit=True, config=args)
    wandb_logger = WandbLogger(project='focus_modeling')
    wandb.watch(model, log_freq=10)


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
        'precision': 32,
        'logger': wandb_logger}


    if args['checkpoint']:
        print(':: Load checkpoint from hparams :')
        print(torch.load(args['checkpoint'])['hyper_parameters'])
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['checkpoint'])

    print(":: Start Training ::")
    trainer = Trainer(**trainer_args)

    trainer.fit(model)
    wandb.finish()


if __name__ == "__main__":
    main()
