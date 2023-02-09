import os, json
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
from datasets import load_metric
from nltk.tokenize import wordpunct_tokenize
from utils import get_data_loaders, add_special_tokens_, special_tokens
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


logger = logging.getLogger(__file__)


def word_level_f1(pred_toks, true_toks):
    eps=1e-10
    # print("pred_toks:", pred_toks)
    prec_list = [1 if word in true_toks else 0 for word in pred_toks]
    prec = sum(prec_list)/len(prec_list)
    rec_list = [1 if word in pred_toks else 0 for word in true_toks]
    rec = sum(rec_list)/len(rec_list)
    f1_score = 2*(prec*rec)/(prec+rec+eps)
    return f1_score


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # kwargs are saved to self.hparams
        self.do_sample = self.hparams.do_sample
        self.num_beams = self.hparams.num_beams
        self.top_k = self.hparams.top_k
        self.no_repeat_ngram_size = self.hparams.no_repeat_ngram_size

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer, BartForConditionalGeneration
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            self.model, self.tokenizer = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
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

        if len(self.hparams.checkpoint) > 0:
            checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
            checkpoint = {k[6:]: v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint)


        train_dataset, valid_dataset = get_data_loaders(self.hparams, self.tokenizer)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        print("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))


    def test_dataloader(self):
        print("\n:: Load and preprocess VALID dataset ::")
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
        valid_loader = DataLoader(self.valid_dataset, sampler=valid_sampler, batch_size=self.hparams.test_batch_size)
        return valid_loader


    def step(self, batch, batch_idx):
        output = self.model(**batch)

        result = {
            'loss':output['loss'] if 'loss' in output else None,
            'logits':output['logits'] if 'logits' in output else None
        }
        return result



    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, knowledge = batch
        # inputs = {'input_ids': input_ids, 'do_sample': self.do_sample, 'num_beams': self.num_beams, 'top_k': self.top_k, 'no_repeat_ngram_size': self.no_repeat_ngram_size}
        inputs = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids, 'labels': lm_labels}

        result = self.step(inputs, batch_idx)
        lm_logits = result['logits']

        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))
        # ppl = torch.exp(lm_loss)
        ppl = torch.exp(result['loss'])

        out_ids = self.model.generate(input_ids=input_ids, do_sample=self.do_sample,
                                      num_beams=self.num_beams, num_return_sequences=1, top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size,
                                      min_length=10)
        output = self.tokenizer.batch_decode(out_ids.tolist(), skip_special_tokens=True)

        result_dict = {}
        result_dict['lm_loss'] = lm_loss.detach()
        result_dict['ppl'] = ppl.detach()
        result_dict['input'] = self.tokenizer.decode(list(input_ids[0]), skip_special_tokens=True)
        # result_dict['input_ids'] = input_ids.cpu().tolist()
        result_dict['output'] = output
        result_dict['labels'] = self.tokenizer.decode(list(decoder_input_ids[0]), skip_special_tokens=True)
        result_dict['knowledge'] = self.tokenizer.decode(list(knowledge[0]), skip_special_tokens=True)
        return result_dict


    def epoch_end(self, outputs, state='test'):
        return outputs
        

    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')

        text_result = []
        f1 = 0
        bleu = 0
        ppl = 0
        bleu_metric = load_metric("sacrebleu")

        for index, i in enumerate(result):
            text_dict = dict()
            ppl += i['ppl'].item()
            pred_sent = i['output']
            true_sent = i['labels']
            pred_toks_list = [wordpunct_tokenize(sent.strip()) for sent in pred_sent]
            true_toks_list = [wordpunct_tokenize(sent.strip()) for sent in true_sent]
            f1 += sum([word_level_f1(pred_toks, true_toks) for pred_toks, true_toks in zip(pred_toks_list, true_toks_list)]) / len(pred_sent)
            bleu += sum([bleu_metric.compute(predictions=[pred], references=[[true]])['score'] for pred, true in zip(pred_sent, true_sent)]) / len(pred_sent)
            text_dict['knowledge'] = i['knowledge']
            text_dict['input'] = i['input']
            text_dict['pred'] = pred_sent
            text_dict['true'] = true_sent
            text_result.append(text_dict)
        
        avg_ppl = ppl / (index+1)
        avg_f1 = f1 / (index+1)
        avg_bleu = bleu / (index+1)

        self.log('test_ppl', avg_ppl)
        self.log('test_f1', avg_f1)
        self.log('test_bleu', avg_bleu)

        result_dict = dict()
        result_dict['ppl'] = avg_ppl
        result_dict['f1'] = avg_f1
        result_dict['bleu'] = avg_bleu
        print(result_dict.items())
        result_dict['text_result'] = text_result

        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')

        result = {
            'ppl': avg_ppl,
            'f1': avg_f1,
            'bleu': avg_bleu
        }

        return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
                        # or "facebook/bart-large"
    parser.add_argument("--checkpoint", type=str, default="/data/yoonna/focusmodeling/wow/model/E5/epoch4-ppl10.6098.ckpt", help="load checkpoint and resume train")
    parser.add_argument("--train_dataset_path", type=str, default="data/train.json", help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_path", type=str, default="data/valid_topic_split.json", help="Path or url of the dataset.")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--num_beams", type=int, default=10, help="number of beams")
    parser.add_argument("--top_k", type=int, default=50, help="top_k ")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--cpu_workers", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--flag", type=str, default="output_beam1_09k", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=19950604)
    parser.add_argument("--output_dir", type=str, default="./output/", help="directory where the model to be saved on")
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

    flag = args['flag']


    trainer_args = {
        'num_sanity_val_steps': 2, #None if args['test_mode'] else 0
        'deterministic': False,
        'gpus': args['gpu_num'],
        'strategy': DDPPlugin(find_unused_parameters=True),
        'precision': 32}



    print(":: Start Training ::")
    trainer = Trainer(**trainer_args)

    model.freeze()
    with torch.no_grad():
        trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    main()
