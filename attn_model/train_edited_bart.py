from setproctitle import setproctitle
setproctitle("leejeongwoo")

from argparse import ArgumentParser
import re
import os
import json
import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import AdamW
from collections import Counter, defaultdict
from data_utils import add_special_tokens_, special_tokens
from datasets import load_metric
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


MODEL_INPUTS = ["input_ids", "decoder_input_ids", "labels", "original"]


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
        # from regeneration_modules import BartEncDec
        # self.config = BartConfig.from_pretrained(self.hparams.pretrained_model)
        self.model = BartForConditionalGeneration.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.pretrained_model)
        self.model.to(self.hparams.device)

        self.test_total_data = 0


        ##### <add special tokens> #####
        #add_special_tokens_(self.model, self.tokenizer, special_tokens=)
        # self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        # e.g., special_tokens = {'subj_token': '<subj>'}
        # orig_num_tokens = len(tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
        self.tokenizer.__dict__.update(special_tokens)
        if num_added_tokens > 0:
            print(num_added_tokens, 'tokens are added!\n')
            self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

        self.persona_st = self.tokenizer.convert_tokens_to_ids(self.tokenizer.persona_token)
        self.knowledge_st = self.tokenizer.convert_tokens_to_ids(self.tokenizer.knowledge_token)
        self.machine_st = self.tokenizer.convert_tokens_to_ids(self.tokenizer.machine_token)


        ##### <parameter mapping> #####
        model_param = dict(self.model.named_parameters())
        
        edited_file_check = 'model.decoder.layers.0.encoder_attn2.k_proj.bias' in model_param.keys()
        print("\n'model.decoder.layers.0.encoder_attn2.k_proj.bias' in model_param.keys():", edited_file_check)
        if (self.hparams.use_edited_bart is True and edited_file_check is False) or (self.hparams.use_edited_bart is False and edited_file_check is True):
            raise Exception(f"\nself.hparams.use_edited_bart: {self.hparams.use_edited_bart}\nedited_file_check: {edited_file_check}")
        
        if 'model.decoder.layers.0.encoder_attn2.k_proj.bias' in model_param.keys():
            base_used_list = ['decoder.layers.0.encoder_attn.k_proj.bias', 'decoder.layers.0.encoder_attn.k_proj.weight', 'decoder.layers.0.encoder_attn.out_proj.bias', 'decoder.layers.0.encoder_attn.out_proj.weight', 'decoder.layers.0.encoder_attn.q_proj.bias', 'decoder.layers.0.encoder_attn.q_proj.weight', 'decoder.layers.0.encoder_attn.v_proj.bias', 'decoder.layers.0.encoder_attn.v_proj.weight',
                            'decoder.layers.1.encoder_attn.k_proj.bias', 'decoder.layers.1.encoder_attn.k_proj.weight', 'decoder.layers.1.encoder_attn.out_proj.bias', 'decoder.layers.1.encoder_attn.out_proj.weight', 'decoder.layers.1.encoder_attn.q_proj.bias', 'decoder.layers.1.encoder_attn.q_proj.weight', 'decoder.layers.1.encoder_attn.v_proj.bias', 'decoder.layers.1.encoder_attn.v_proj.weight',
                            'decoder.layers.2.encoder_attn.k_proj.bias', 'decoder.layers.2.encoder_attn.k_proj.weight', 'decoder.layers.2.encoder_attn.out_proj.bias', 'decoder.layers.2.encoder_attn.out_proj.weight', 'decoder.layers.2.encoder_attn.q_proj.bias', 'decoder.layers.2.encoder_attn.q_proj.weight', 'decoder.layers.2.encoder_attn.v_proj.bias', 'decoder.layers.2.encoder_attn.v_proj.weight',
                            'decoder.layers.3.encoder_attn.k_proj.bias', 'decoder.layers.3.encoder_attn.k_proj.weight', 'decoder.layers.3.encoder_attn.out_proj.bias', 'decoder.layers.3.encoder_attn.out_proj.weight', 'decoder.layers.3.encoder_attn.q_proj.bias', 'decoder.layers.3.encoder_attn.q_proj.weight', 'decoder.layers.3.encoder_attn.v_proj.bias', 'decoder.layers.3.encoder_attn.v_proj.weight',
                            'decoder.layers.4.encoder_attn.k_proj.bias', 'decoder.layers.4.encoder_attn.k_proj.weight', 'decoder.layers.4.encoder_attn.out_proj.bias', 'decoder.layers.4.encoder_attn.out_proj.weight', 'decoder.layers.4.encoder_attn.q_proj.bias', 'decoder.layers.4.encoder_attn.q_proj.weight', 'decoder.layers.4.encoder_attn.v_proj.bias', 'decoder.layers.4.encoder_attn.v_proj.weight',
                            'decoder.layers.5.encoder_attn.k_proj.bias', 'decoder.layers.5.encoder_attn.k_proj.weight', 'decoder.layers.5.encoder_attn.out_proj.bias', 'decoder.layers.5.encoder_attn.out_proj.weight', 'decoder.layers.5.encoder_attn.q_proj.bias', 'decoder.layers.5.encoder_attn.q_proj.weight', 'decoder.layers.5.encoder_attn.v_proj.bias', 'decoder.layers.5.encoder_attn.v_proj.weight']
            base_used_list.sort()
            print("len(base_used_list):", len(base_used_list))

            new_init_list = ['decoder.layers.0.encoder_attn2.k_proj.bias', 'decoder.layers.0.encoder_attn2.k_proj.weight', 'decoder.layers.0.encoder_attn2.out_proj.bias', 'decoder.layers.0.encoder_attn2.out_proj.weight', 'decoder.layers.0.encoder_attn2.q_proj.bias', 'decoder.layers.0.encoder_attn2.q_proj.weight', 'decoder.layers.0.encoder_attn2.v_proj.bias', 'decoder.layers.0.encoder_attn2.v_proj.weight',
                            'decoder.layers.0.encoder_attn3.k_proj.bias', 'decoder.layers.0.encoder_attn3.k_proj.weight', 'decoder.layers.0.encoder_attn3.out_proj.bias', 'decoder.layers.0.encoder_attn3.out_proj.weight', 'decoder.layers.0.encoder_attn3.q_proj.bias', 'decoder.layers.0.encoder_attn3.q_proj.weight', 'decoder.layers.0.encoder_attn3.v_proj.bias', 'decoder.layers.0.encoder_attn3.v_proj.weight',
                            'decoder.layers.0.encoder_attn4.k_proj.bias', 'decoder.layers.0.encoder_attn4.k_proj.weight', 'decoder.layers.0.encoder_attn4.out_proj.bias', 'decoder.layers.0.encoder_attn4.out_proj.weight', 'decoder.layers.0.encoder_attn4.q_proj.bias', 'decoder.layers.0.encoder_attn4.q_proj.weight', 'decoder.layers.0.encoder_attn4.v_proj.bias', 'decoder.layers.0.encoder_attn4.v_proj.weight',
                            'decoder.layers.0.encoder_attn5.k_proj.bias', 'decoder.layers.0.encoder_attn5.k_proj.weight', 'decoder.layers.0.encoder_attn5.out_proj.bias', 'decoder.layers.0.encoder_attn5.out_proj.weight', 'decoder.layers.0.encoder_attn5.q_proj.bias', 'decoder.layers.0.encoder_attn5.q_proj.weight', 'decoder.layers.0.encoder_attn5.v_proj.bias', 'decoder.layers.0.encoder_attn5.v_proj.weight',
                            'decoder.layers.1.encoder_attn2.k_proj.bias', 'decoder.layers.1.encoder_attn2.k_proj.weight', 'decoder.layers.1.encoder_attn2.out_proj.bias', 'decoder.layers.1.encoder_attn2.out_proj.weight', 'decoder.layers.1.encoder_attn2.q_proj.bias', 'decoder.layers.1.encoder_attn2.q_proj.weight', 'decoder.layers.1.encoder_attn2.v_proj.bias', 'decoder.layers.1.encoder_attn2.v_proj.weight',
                            'decoder.layers.1.encoder_attn3.k_proj.bias', 'decoder.layers.1.encoder_attn3.k_proj.weight', 'decoder.layers.1.encoder_attn3.out_proj.bias', 'decoder.layers.1.encoder_attn3.out_proj.weight', 'decoder.layers.1.encoder_attn3.q_proj.bias', 'decoder.layers.1.encoder_attn3.q_proj.weight', 'decoder.layers.1.encoder_attn3.v_proj.bias', 'decoder.layers.1.encoder_attn3.v_proj.weight',
                            'decoder.layers.1.encoder_attn4.k_proj.bias', 'decoder.layers.1.encoder_attn4.k_proj.weight', 'decoder.layers.1.encoder_attn4.out_proj.bias', 'decoder.layers.1.encoder_attn4.out_proj.weight', 'decoder.layers.1.encoder_attn4.q_proj.bias', 'decoder.layers.1.encoder_attn4.q_proj.weight', 'decoder.layers.1.encoder_attn4.v_proj.bias', 'decoder.layers.1.encoder_attn4.v_proj.weight',
                            'decoder.layers.1.encoder_attn5.k_proj.bias', 'decoder.layers.1.encoder_attn5.k_proj.weight', 'decoder.layers.1.encoder_attn5.out_proj.bias', 'decoder.layers.1.encoder_attn5.out_proj.weight', 'decoder.layers.1.encoder_attn5.q_proj.bias', 'decoder.layers.1.encoder_attn5.q_proj.weight', 'decoder.layers.1.encoder_attn5.v_proj.bias', 'decoder.layers.1.encoder_attn5.v_proj.weight',
                            'decoder.layers.2.encoder_attn2.k_proj.bias', 'decoder.layers.2.encoder_attn2.k_proj.weight', 'decoder.layers.2.encoder_attn2.out_proj.bias', 'decoder.layers.2.encoder_attn2.out_proj.weight', 'decoder.layers.2.encoder_attn2.q_proj.bias', 'decoder.layers.2.encoder_attn2.q_proj.weight', 'decoder.layers.2.encoder_attn2.v_proj.bias', 'decoder.layers.2.encoder_attn2.v_proj.weight',
                            'decoder.layers.2.encoder_attn3.k_proj.bias', 'decoder.layers.2.encoder_attn3.k_proj.weight', 'decoder.layers.2.encoder_attn3.out_proj.bias', 'decoder.layers.2.encoder_attn3.out_proj.weight', 'decoder.layers.2.encoder_attn3.q_proj.bias', 'decoder.layers.2.encoder_attn3.q_proj.weight', 'decoder.layers.2.encoder_attn3.v_proj.bias', 'decoder.layers.2.encoder_attn3.v_proj.weight',
                            'decoder.layers.2.encoder_attn4.k_proj.bias', 'decoder.layers.2.encoder_attn4.k_proj.weight', 'decoder.layers.2.encoder_attn4.out_proj.bias', 'decoder.layers.2.encoder_attn4.out_proj.weight', 'decoder.layers.2.encoder_attn4.q_proj.bias', 'decoder.layers.2.encoder_attn4.q_proj.weight', 'decoder.layers.2.encoder_attn4.v_proj.bias', 'decoder.layers.2.encoder_attn4.v_proj.weight',
                            'decoder.layers.2.encoder_attn5.k_proj.bias', 'decoder.layers.2.encoder_attn5.k_proj.weight', 'decoder.layers.2.encoder_attn5.out_proj.bias', 'decoder.layers.2.encoder_attn5.out_proj.weight', 'decoder.layers.2.encoder_attn5.q_proj.bias', 'decoder.layers.2.encoder_attn5.q_proj.weight', 'decoder.layers.2.encoder_attn5.v_proj.bias', 'decoder.layers.2.encoder_attn5.v_proj.weight',
                            'decoder.layers.3.encoder_attn2.k_proj.bias', 'decoder.layers.3.encoder_attn2.k_proj.weight', 'decoder.layers.3.encoder_attn2.out_proj.bias', 'decoder.layers.3.encoder_attn2.out_proj.weight', 'decoder.layers.3.encoder_attn2.q_proj.bias', 'decoder.layers.3.encoder_attn2.q_proj.weight', 'decoder.layers.3.encoder_attn2.v_proj.bias', 'decoder.layers.3.encoder_attn2.v_proj.weight',
                            'decoder.layers.3.encoder_attn3.k_proj.bias', 'decoder.layers.3.encoder_attn3.k_proj.weight', 'decoder.layers.3.encoder_attn3.out_proj.bias', 'decoder.layers.3.encoder_attn3.out_proj.weight', 'decoder.layers.3.encoder_attn3.q_proj.bias', 'decoder.layers.3.encoder_attn3.q_proj.weight', 'decoder.layers.3.encoder_attn3.v_proj.bias', 'decoder.layers.3.encoder_attn3.v_proj.weight',
                            'decoder.layers.3.encoder_attn4.k_proj.bias', 'decoder.layers.3.encoder_attn4.k_proj.weight', 'decoder.layers.3.encoder_attn4.out_proj.bias', 'decoder.layers.3.encoder_attn4.out_proj.weight', 'decoder.layers.3.encoder_attn4.q_proj.bias', 'decoder.layers.3.encoder_attn4.q_proj.weight', 'decoder.layers.3.encoder_attn4.v_proj.bias', 'decoder.layers.3.encoder_attn4.v_proj.weight',
                            'decoder.layers.3.encoder_attn5.k_proj.bias', 'decoder.layers.3.encoder_attn5.k_proj.weight', 'decoder.layers.3.encoder_attn5.out_proj.bias', 'decoder.layers.3.encoder_attn5.out_proj.weight', 'decoder.layers.3.encoder_attn5.q_proj.bias', 'decoder.layers.3.encoder_attn5.q_proj.weight', 'decoder.layers.3.encoder_attn5.v_proj.bias', 'decoder.layers.3.encoder_attn5.v_proj.weight',
                            'decoder.layers.4.encoder_attn2.k_proj.bias', 'decoder.layers.4.encoder_attn2.k_proj.weight', 'decoder.layers.4.encoder_attn2.out_proj.bias', 'decoder.layers.4.encoder_attn2.out_proj.weight', 'decoder.layers.4.encoder_attn2.q_proj.bias', 'decoder.layers.4.encoder_attn2.q_proj.weight', 'decoder.layers.4.encoder_attn2.v_proj.bias', 'decoder.layers.4.encoder_attn2.v_proj.weight',
                            'decoder.layers.4.encoder_attn3.k_proj.bias', 'decoder.layers.4.encoder_attn3.k_proj.weight', 'decoder.layers.4.encoder_attn3.out_proj.bias', 'decoder.layers.4.encoder_attn3.out_proj.weight', 'decoder.layers.4.encoder_attn3.q_proj.bias', 'decoder.layers.4.encoder_attn3.q_proj.weight', 'decoder.layers.4.encoder_attn3.v_proj.bias', 'decoder.layers.4.encoder_attn3.v_proj.weight',
                            'decoder.layers.4.encoder_attn4.k_proj.bias', 'decoder.layers.4.encoder_attn4.k_proj.weight', 'decoder.layers.4.encoder_attn4.out_proj.bias', 'decoder.layers.4.encoder_attn4.out_proj.weight', 'decoder.layers.4.encoder_attn4.q_proj.bias', 'decoder.layers.4.encoder_attn4.q_proj.weight', 'decoder.layers.4.encoder_attn4.v_proj.bias', 'decoder.layers.4.encoder_attn4.v_proj.weight',
                            'decoder.layers.4.encoder_attn5.k_proj.bias', 'decoder.layers.4.encoder_attn5.k_proj.weight', 'decoder.layers.4.encoder_attn5.out_proj.bias', 'decoder.layers.4.encoder_attn5.out_proj.weight', 'decoder.layers.4.encoder_attn5.q_proj.bias', 'decoder.layers.4.encoder_attn5.q_proj.weight', 'decoder.layers.4.encoder_attn5.v_proj.bias', 'decoder.layers.4.encoder_attn5.v_proj.weight',
                            'decoder.layers.5.encoder_attn2.k_proj.bias', 'decoder.layers.5.encoder_attn2.k_proj.weight', 'decoder.layers.5.encoder_attn2.out_proj.bias', 'decoder.layers.5.encoder_attn2.out_proj.weight', 'decoder.layers.5.encoder_attn2.q_proj.bias', 'decoder.layers.5.encoder_attn2.q_proj.weight', 'decoder.layers.5.encoder_attn2.v_proj.bias', 'decoder.layers.5.encoder_attn2.v_proj.weight',
                            'decoder.layers.5.encoder_attn3.k_proj.bias', 'decoder.layers.5.encoder_attn3.k_proj.weight', 'decoder.layers.5.encoder_attn3.out_proj.bias', 'decoder.layers.5.encoder_attn3.out_proj.weight', 'decoder.layers.5.encoder_attn3.q_proj.bias', 'decoder.layers.5.encoder_attn3.q_proj.weight', 'decoder.layers.5.encoder_attn3.v_proj.bias', 'decoder.layers.5.encoder_attn3.v_proj.weight',
                            'decoder.layers.5.encoder_attn4.k_proj.bias', 'decoder.layers.5.encoder_attn4.k_proj.weight', 'decoder.layers.5.encoder_attn4.out_proj.bias', 'decoder.layers.5.encoder_attn4.out_proj.weight', 'decoder.layers.5.encoder_attn4.q_proj.bias', 'decoder.layers.5.encoder_attn4.q_proj.weight', 'decoder.layers.5.encoder_attn4.v_proj.bias', 'decoder.layers.5.encoder_attn4.v_proj.weight',
                            'decoder.layers.5.encoder_attn5.k_proj.bias', 'decoder.layers.5.encoder_attn5.k_proj.weight', 'decoder.layers.5.encoder_attn5.out_proj.bias', 'decoder.layers.5.encoder_attn5.out_proj.weight', 'decoder.layers.5.encoder_attn5.q_proj.bias', 'decoder.layers.5.encoder_attn5.q_proj.weight', 'decoder.layers.5.encoder_attn5.v_proj.bias', 'decoder.layers.5.encoder_attn5.v_proj.weight']
            new_init_list.sort()
            print("len(new_init_list):", len(new_init_list))

            error_count = 0
            for each_param in new_init_list:
                param_name = each_param.replace("encoder_attn2", "encoder_attn")
                param_name = param_name.replace("encoder_attn3", "encoder_attn")
                param_name = param_name.replace("encoder_attn4", "encoder_attn")
                param_name = param_name.replace("encoder_attn5", "encoder_attn")

                assert "model." + each_param in model_param.keys()
                if param_name in base_used_list:
                    model_param["model." + each_param].data.copy_(model_param["model." + param_name].data)      # [새 모델].data.copy_([기존 모델])
                else:
                    error_count += 1
                    raise Exception("parameter mapping error")
            print(f"\nparameter mapping error_count: {error_count}\n")





    def build_input(self, knowledge, persona, before_refine, label, original):
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        input_ids = [bos] + [self.knowledge_st] + knowledge + [self.persona_st] + persona + [self.machine_st] + before_refine + [eos]

        instance = dict()
        instance['input_ids'] = input_ids
        instance['decoder_input_ids'] = [bos] + label
        instance['labels'] = label + [eos]
        instance['original'] = original

        return instance

    def pad_dataset(self, dataset, padding):
        #breakpoint()
        #len_list_input  =
        #len_list_output =
        max_l = max(len(x) for x in dataset["input_ids"])
        max_dec_l = max(len(x) for x in dataset["decoder_input_ids"])
        max_origin_l = max(len(x) for x in dataset["original"])
        dataset['input_ids'] = [x + [padding] * (max_l - len(x)) for x in dataset['input_ids']]
        dataset['decoder_input_ids'] = [x + [padding] * ((max_dec_l) - len(x)) for x in dataset['decoder_input_ids']]
        dataset['labels'] = [x + [-100] * (max_dec_l - len(x)) for x in dataset['labels']]
        dataset['original'] = [x + [padding] * (max_origin_l - len(x)) for x in dataset['original']]
        return dataset

    def dataloader(self, type_name):
        if type_name in ["train", "valid"]:
            from utils_focus import get_dataset_refine_edited_bart
            refine_data = get_dataset_refine_edited_bart(self.tokenizer, train_dataset_path=self.hparams.train_dataset_path, train_dataset_cache=self.hparams.train_dataset_cache, dev_dataset_path=self.hparams.dev_dataset_path, dev_dataset_cache=self.hparams.dev_dataset_cache)
            print("\n\nBuild inputs and labels (train_data, valid_data)")
            datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        elif type_name in ["test"]:
            from utils_focus import get_dataset_refine_edited_bart_test
            refine_data = get_dataset_refine_edited_bart_test(self.tokenizer, test_dataset_path=self.hparams.test_dataset_path, test_dataset_cache=self.hparams.test_dataset_cache)
            print("Build inputs and labels (test_data)")
            datasets = {"test": defaultdict(list)}
        else:
            raise Exception("type_name error 1")

        for (key, value) in refine_data.items():
            for data in value:      # ["knowledge", "persona", "before_refine", "label", "input_text", "input_ids_split_decoded"]
                knowledge = data["knowledge"]
                persona = data['persona']
                before_refine = data['before_refine']
                label = data['label']
                input_text = data['input_text']
                input_ids_split_decoded = data['input_ids_split_decoded']

                instance = self.build_input(knowledge, persona, before_refine, label, label)
                for input_name, input_array in instance.items():
                    datasets[key][input_name].append(input_array)

        print("Pad inputs and convert to Tensor")
        if type_name in ["train", "valid"]:
            tensor_datasets = {"train": [], "valid": []}
        elif type_name in ["test"]:
            tensor_datasets = {"test": []}
        else:
            raise Exception("type_name error 2")
        
        for dataset_name, dataset in datasets.items():
            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_token_id)
            for input_name in MODEL_INPUTS:
                tensor = torch.tensor(dataset[input_name])
                # print(input_name, tensor.size())
                tensor_datasets[dataset_name].append(tensor)

        if type_name in ["train", "valid"]:
            print("Build train and valid dataloaders")
            train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
            return train_dataset, valid_dataset
        elif type_name in ["test"]:
            print("Build test dataloaders")
            test_dataset = TensorDataset(*tensor_datasets["test"])
            return test_dataset
        else:
            raise Exception("type_name error 3")

    def train_dataloader(self):
        train_dataset, _ = self.dataloader(type_name="train")
        print("\nTrain dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        _, valid_dataset = self.dataloader(type_name="valid")
        print("\nValid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))
        return DataLoader(valid_dataset, batch_size=self.hparams.valid_batch_size, shuffle=False)

    def test_dataloader(self):
        test_dataset = self.dataloader(type_name="test")
        print("\nTest dataset (Batch, Seq length): {}".format(test_dataset.tensors[0].shape))
        self.test_total_data = len(test_dataset)
        print("self.test_total_data:", self.test_total_data)
        return DataLoader(test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)





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

    def step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels, original = batch
        output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        lm_loss = result['loss']
        loss = lm_loss / self.hparams.grad_accum
        self.log('train_loss', loss)
        result['loss'] = loss
        return result

    # sanity checking에 대한 참고자료(https://blog-deepest.medium.com/pytorch-lightning-%EC%B0%8D%EB%A8%B9-%ED%9B%84%EA%B8%B0-1dc9bef69527)
    def validation_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        #print(result.items())
        result = {k: v.detach().cpu() for k, v in result.items()}   # cpu()로 해줘야 gpu memory가 쌓이지 않음
        self.log('valid_loss', result['loss'])
        return result

    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels, original = batch
        # print("input_ids:", input_ids.size())
        # print("batch_idx:", batch_idx)
        # print("labels:", labels.size())
        # print("original:", original.size())
        # print("test_step - decoder_input_ids:", decoder_input_ids.size())
        before_refine = input_ids
        with torch.no_grad():
            output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)

            mask = (input_ids != self.tokenizer.pad_token_id)
            input_ids = input_ids[mask]
            # print("input_ids:", input_ids.size())
            # breakpoint()
            gen_output = self.model.generate(input_ids.unsqueeze(0))    #, num_beams=1)

        result = {"before_refine": before_refine, 'gen_output': gen_output, 'gold_utterance': original, 'for_ppl_result': {k: v.detach().cpu() for k, v in output.items()}}
        return result


    def epoch_end(self, outputs, state='train'):
        if state=='train' or state=='val' or state=='test':
            loss = torch.tensor(0, dtype=torch.float).to(self.hparams.device)
            ppl = torch.tensor(0, dtype=torch.float).to(self.hparams.device)

            for i in outputs:
                loss += i['loss']
                ppl += torch.exp(i['loss'])

            loss = loss / len(outputs)
            ppl = ppl / len(outputs)

            result = {'loss': loss, 'ppl': ppl}
        return result

    def train_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='train')
        self.log('train_loss', result['loss'])
        self.log('train_ppl', result['ppl'])
        return result

    def validation_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='val')
        self.log('valid_loss', result['loss'])
        self.log('valid_ppl', result['ppl'])
        return result

    def test_epoch_end(self, outputs):
        result = self.epoch_end([each_output['for_ppl_result'] for each_output in outputs], state='test')
        self.log('test_ppl', result['ppl'])

        bleu_ori, bleu_ref = 0, 0
        bleu_metric = load_metric("sacrebleu")
        refine_total = 0
        result_list = list()

        for test_data in outputs:
            output_dict = dict()
            # print("\ntest_data['before_refine']:", test_data['before_refine'])
            # print("\ntest_data['gen_output']:", test_data['gen_output'])
            # print("\ntest_data['gold_utterance']:", test_data['gold_utterance'])
            before_refine = self.tokenizer.decode(test_data['before_refine'].tolist()[0], skip_special_tokens=True)
            pred_reply = self.tokenizer.decode(test_data['gen_output'].tolist()[0], skip_special_tokens=True)
            gold_reply = self.tokenizer.decode(test_data['gold_utterance'].tolist()[0], skip_special_tokens=True)

            refine_total += 1
            output_dict['before_refine'] = before_refine
            output_dict['after_refine'] = pred_reply
            output_dict['gold_utterance'] = gold_reply
            result_list.append(output_dict)

            #sacre BLEU
            bleu_ref += bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']
            bleu_ori += bleu_metric.compute(predictions=[before_refine], references=[[gold_reply]])['score']

            print("\nbefore refine: ", before_refine)
            print("after_refine: ", pred_reply)
            print("gold_utterance: ", gold_reply)
            
        bleu_result_ref = bleu_ref/refine_total
        bleu_result_ori = bleu_ori/refine_total

        result_dict = dict()
        result_dict['ppl'] = result['ppl'].item()
        result_dict['bleu_ori'] = bleu_result_ori
        result_dict['bleu_ref'] = bleu_result_ref
        result_dict['refine_num'] = refine_total
        self.log('test_bleu_ori', bleu_result_ori)
        self.log('test_bleu_ref', bleu_result_ref)
        self.log('test_total_refine_num', int(refine_total))
        print(result_dict.items())
        
        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list
        with open(self.hparams.output_dir + 'result_dict.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')
        return test_result



def main():
    now = datetime.datetime.now()
    output_dir_name = now.strftime("%Y%m%d_%H%M%S")
    # output_dir_name = f"{t.year}{t.month}{t.day}_{t.hour}{t.minute}{t.second}"
    if not os.path.exists(f"./regen/{output_dir_name}"):
        os.makedirs(f"./regen/{output_dir_name}")

    parser = ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="./our_refine_data/wow/wow_refine_train_data.json", help="regen train dataset path")
    parser.add_argument("--train_dataset_cache", type=str, default="cache.tar.gz", help="regen train datacache path")
    parser.add_argument("--dev_dataset_path", type=str, default="./our_refine_data/wow/wow_refine_valid_data.json", help="regen valid dataset path")
    parser.add_argument("--dev_dataset_cache", type=str, default="cache.tar.gz", help="regen valid datacache path")
    parser.add_argument("--test_dataset_path", type=str, default="./our_refine_data/wow/wow_refine_test_data.json", help="regen test dataset path")
    parser.add_argument("--test_dataset_cache", type=str, default="cache.tar.gz", help="regen test datacache path")

    parser.add_argument("--use_edited_bart", action='store_true')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    # parser.add_argument("--ckpt", type=str, default="facebook/bart-base", help="ckpt path") #facebook/bart-base
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=951014)
    # parser.add_argument("--lm_coef", type=float, default=1.0, help="Coefficient for LM loss")
    # parser.add_argument("--cls_coef", type=float, default=1.0, help="Coefficient for CLS loss")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="{AdamW, AdamP}")
    parser.add_argument("--lr_scheduler", type=str, default="lambdalr", help="{exp, lambdalr}")
    parser.add_argument("--grad_accum", type=int, default=32, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--precision", type=int, default=32, help="{16,32,64}")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default=f"./regen/{output_dir_name}/", help="default value for PLMs")
    args = vars(parser.parse_args())
    
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])

    model = Model(**args)
    model.train()       # 원래 model.eval()로 되어있던데 왜 model.eval()였는지는 모르겠음
    model.model.train()
    model.to('cuda')

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['output_dir'],
        filename='epoch{epoch}-valid_loss{valid_loss:.4f}',
        monitor='valid_loss',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor='valid_loss',
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
        # 'strategy': DDPPlugin(find_unused_parameters=True),
        'precision': args['precision'],
        'logger': wandb_logger}

    # if args['ckpt'] not in ['facebook/bart-base']:
    #     print(':: Load checkpoint from hparams ::')
    #     print(torch.load(args['ckpt'])['hyper_parameters'])
    #     trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['ckpt'])

    trainer = Trainer(**trainer_args)
    trainer.fit(model)



    print(":: Start Testing ::")
    model.eval()
    model.model.eval()
    model.freeze()
    with torch.no_grad():
        trainer.test(model)
    wandb.finish()


if __name__ == "__main__":
    main()