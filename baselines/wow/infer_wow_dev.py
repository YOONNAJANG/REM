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
from utils import get_data_loaders, add_special_tokens_, special_tokens
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


logger = logging.getLogger(__file__)

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
        print("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))


    def test_dataloader(self):
        print("\n:: Load and preprocess VALID dataset ::")
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
        valid_loader = DataLoader(self.valid_dataset, sampler=valid_sampler, batch_size=self.hparams.test_batch_size)
        return valid_loader


    def step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model.generate(**batch)
        result = {
            'output':output
        }
        return result



    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, persona = batch
        inputs = {'input_ids': input_ids, 'do_sample': self.do_sample, 'num_beams': self.num_beams, 'top_k': self.top_k, 'no_repeat_ngram_size': self.no_repeat_ngram_size}
        result = self.step(inputs, batch_idx)
        output = self.tokenizer.decode(list(result['output'][0]), skip_special_tokens=True)

        result_dict = {}
        result_dict['output'] = output
        result_dict['input'] = self.tokenizer.decode(list(input_ids[0]), skip_special_tokens=True)
        result_dict['input_ids'] = input_ids.cpu().tolist()
        result_dict['labels'] = self.tokenizer.decode(list(decoder_input_ids[0]), skip_special_tokens=True)
        result_dict['persona'] =self.tokenizer.decode(list(persona[0]), skip_special_tokens=True)
        return result_dict



    def epoch_end(self, outputs):
        # result_dict = {}
        # for i in outputs:
        #     output = i['output']
        #     input_ids = i['input_ids']
        #     labels = i['labels']
        #     input = i['input']
        result_dict = {}
        result_dict['data'] = outputs
        # breakpoint()
        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')
        return result_dict

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BART",
                        help="{BART, T5, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                    help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")  # or "facebook/bart-large"
    parser.add_argument("--checkpoint", type=str, default="/home/data/yoonna/focusmodeling/wow/model/E5/epoch4-ppl10.6098.ckpt", help="load checkpoint and resume train")
    parser.add_argument("--train_dataset_path", type=str, default="/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_path", type=str, default="/home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beams")
    parser.add_argument("--top_k", type=int, default=50, help="top_k ")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--cpu_workers", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--flag", type=str, default="output_beam1_09k", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=19950604)
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/eval_output/wow/output/", help="directory where the model to be saved on")
    args = vars(parser.parse_args())

    print(":: Using PyTorch Ver", torch.__version__, " ::")
    print(":: Fix Seed", args['seed'], " ::")
    from setproctitle import setproctitle
    setproctitle("suhyun")

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
