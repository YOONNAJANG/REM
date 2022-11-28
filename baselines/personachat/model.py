import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule
from torch.nn import Softmax
from datasets import load_metric
from nltk.tokenize import wordpunct_tokenize
from utils import add_special_tokens_, special_tokens, get_data_loaders, get_testdata_loaders
import json
from ptuning import get_embedding_layer, PromptEncoder, get_vocab_by_strategy

def word_level_f1(pred_toks, true_toks):
    eps=1e-10
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
        self.pseudo_token = self.hparams.pseudo_token
        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer, BartForConditionalGeneration
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)



        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
            self.model.to(self.hparams.device)
            self.model.eval()
            #if self.hparams.model_path == "t5-base" or "t5-large":
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        else:
            raise NotImplementedError


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

        train_dataset, valid_dataset = get_data_loaders(self.hparams, self.tokenizer)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        print("Train dataset (Examples, Seq length): {}".format(train_dataset.tensors[0].shape))


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


    def train_dataloader(self):
        print("\n:: Load and preprocess TRAIN dataset ::")
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        train_loader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.hparams.train_batch_size)

        return train_loader

    def val_dataloader(self):
        print("\n:: Load and preprocess VALID dataset ::")
        #valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.hparams.valid_batch_size)
        return valid_loader



    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
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
        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(batch['input_ids'])
            if 'input_ids' in batch:
                del batch['input_ids']
                batch['inputs_embeds'] = input_embeds
            output = self.model(**batch)
        else:
            output = self.model(**batch)

        result = {
            'loss':output['loss'] if 'loss' in output else None,
            'logits':output['logits'] if 'logits' in output else None
        }
        return result

    def forward(self, batch, batch_idx):
        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(batch['input_ids'])
            if 'input_ids' in batch:
                del batch['input_ids']
                batch['inputs_embeds'] = input_embeds
            output = self.model(**batch)
        else:
            output = self.model(**batch)

        result = {'logits':output['logits'] if 'logits' in output else None}

        return result


    def training_step(self, batch, batch_idx):

        input_ids, decoder_input_ids, lm_labels = batch
        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids,
            'labels':lm_labels
        }
        result = self.step(inputs, batch_idx)

        loss = result['loss']
        ppl = torch.exp(loss)

        self.log('train_ppl', ppl)
        self.log('train_loss', loss)

        result_dict = {
            'loss': loss
        }
        return result_dict


    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels = batch
        inputs = {
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
        }
        result = self.forward(inputs, batch_idx)
        lm_logits = result['logits']

        softmax = Softmax(dim=-1)
        lm_pred = softmax(lm_logits)
        lm_val, lm_pred_idx = torch.topk(lm_pred, k=1, dim=-1)
        lm_pred_idx = lm_pred_idx.squeeze(-1)

        mask = (lm_labels != -100)
        lm_labels_only = [lm_labels[mask].tolist()]
        lm_pred_idx = lm_pred_idx[mask].tolist()

        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = lm_criterion(lm_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))

        ppl = torch.exp(loss)

        result = {
            'loss':loss.detach(),
            'ppl':ppl.detach(),
            'pred':lm_pred_idx,
            'target':lm_labels_only
        }
        return result


    def epoch_end(self, outputs, state='train'):
        if state=='train':
            loss = torch.tensor(0, dtype=torch.float)
            for i in outputs:
                loss += i['loss'].cpu().detach()
            loss = loss / len(outputs)
            ppl = torch.exp(loss)
            result={
                'loss': loss,
                'ppl': ppl
            }

        elif state=='val':
            loss = torch.tensor(0, dtype=torch.float)
            f1 = torch.tensor(0, dtype=torch.float)

            for i in outputs:
                loss += i['loss'].cpu().detach()
            loss = loss / len(outputs)
            ppl = torch.exp(loss)

            self.log('valid_loss', loss)
            self.log('valid_ppl', ppl)
            self.log('valid_pg_f1', f1)

            print('\n\nvalid_loss:', loss)
            print('valid_ppl:', ppl)
            #print('valid_pg_f1:', f1)



            result={
                'ppl': ppl
            }

        return result

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')


class Model_Eval(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # kwargs are saved to self.hparams
        self.pseudo_token = self.hparams.pseudo_token

        if self.hparams.model_name == 'BART':
            from transformers import BartTokenizer, BartForConditionalGeneration
            self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_path)
            self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_path)
            #if self.hparams.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        elif self.hparams.model_name == 'T5':
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_path)
            #if self.hparams.model_path == "t5-base" or "t5-large":
            self.tokenizer, self.model = add_special_tokens_(self.model, self.tokenizer, special_tokens)

        else:
            raise NotImplementedError

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        if len(self.hparams.checkpoint) > 0:
            print("Load weights from ", self.hparams.checkpoint)

            #LM weights load
            checkpoint = torch.load(self.hparams.checkpoint)['module']
            lm_checkpoint = dict()
            prompt_checkpoint = dict()
            else_checkpoint = dict()
            for k,v in checkpoint.items():
                if k.startswith('module.model.') == True:
                    lm_checkpoint[k] = v
                elif k.startswith('module.prompt_encoder.') == True:
                    prompt_checkpoint[k] = v
                else:
                    else_checkpoint[k] = v
            lm_checkpoint = {k[13:]: v for k, v in lm_checkpoint.items()}
            self.model.load_state_dict(lm_checkpoint)

            if self.hparams.ptuning == True:

                self.embeddings = get_embedding_layer(self.hparams, self.model)
                # set allowed vocab set
                self.vocab = self.tokenizer.get_vocab()
                self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.hparams, self.tokenizer))
                self.template = tuple([int(item) for item in self.hparams.template.split(',')])
                # load prompt encoder
                self.hidden_size = self.embeddings.embedding_dim
                self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pseudo_token]})
                self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
                self.spell_length = sum(self.template)
                self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer,
                                                    self.hparams.device, self.hparams)
                prompt_checkpoint = {k[22:]: v for k, v in prompt_checkpoint.items()}
                self.prompt_encoder.load_state_dict(prompt_checkpoint)
                self.prompt_encoder = self.prompt_encoder.to(self.hparams.device)
            #Prompt weight load


        self.model.to(self.hparams.device)
        self.model.eval()

        test_dataset = get_testdata_loaders(self.hparams, self.tokenizer)
        self.test_dataset = test_dataset
        print("Test dataset (Examples, Seq length): {}".format(test_dataset.tensors[0].shape))
        self.num_beams = self.hparams.num_beams
        self.num_return_sequences = self.hparams.num_return_sequences
        self.top_k = self.hparams.top_k
        self.do_sample = self.hparams.do_sample
        self.min_length = self.hparams.min_length
        self.max_length = self.hparams.max_length


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


    def test_dataloader(self):
        print("\n:: Load and preprocess TEST dataset ::")
        #valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset)
        test_loader = DataLoader(self.test_dataset, batch_size=self.hparams.test_batch_size)

        return test_loader

    # def forward(self, **kwargs):
    #     result = self.model(**kwargs)
    #     return result

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
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
            'logits':output['logits'] if 'logits' in output else None
        }
        return result




    def test_step(self, batch, batch_idx):

        input_ids, decoder_input_ids, lm_labels = batch

        inputs = {
            'input_ids':input_ids,
            'decoder_input_ids':decoder_input_ids
        }

        if self.hparams.ptuning == True:
            input_embeds = self.embed_inputs(inputs['input_ids'])
            if 'input_ids' in inputs.keys():
                del inputs['input_ids']
                inputs['inputs_embeds'] = input_embeds
        result = self.step(inputs, batch_idx)
        lm_logits = result['logits']

        #pad_mask = (input_ids != self.tokenizer.pad_token_id)
        #input_ids = input_ids[pad_mask]


        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits.view(-1, self.model.config.vocab_size), lm_labels.view(-1))
        ppl = torch.exp(lm_loss)

        if self.hparams.ptuning == True:
            out_ids = self.model.generate(inputs_embeds=input_embeds, do_sample=self.do_sample,
                                          num_beams=self.num_beams, top_k=self.top_k, no_repeat_ngram_size=2,
                                          num_return_sequences=self.num_return_sequences,
                                          max_length=self.max_length, min_length=self.min_length)

        else:
            out_ids = self.model.generate(input_ids=input_ids, do_sample=self.do_sample,
                                          num_beams=self.num_beams, top_k=self.top_k, no_repeat_ngram_size=2,
                                          num_return_sequences=self.num_return_sequences,
                                          max_length=self.max_length, min_length=self.min_length)

        out_ids = self.tokenizer.batch_decode(out_ids.tolist(), skip_special_tokens=True)


        #lm_masks = (lm_labels != -100)

        lm_labels[lm_labels==-100] = self.pad_token_id
        lm_labels[lm_labels==50266] = self.pad_token_id
        lm_labels = self.tokenizer.batch_decode(lm_labels.tolist(), skip_special_tokens=True)
        #print(f'ppl: {ppl}, output: {out_ids}')

        result = {
            'lm_loss':lm_loss.detach(),
            'ppl':ppl.detach(),
            'model_result': out_ids,
            'labels': lm_labels
        }
        return result


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
            pred_sent = i['model_result']
            true_sent = i['labels']
            pred_toks_list = [wordpunct_tokenize(sent.strip()) for sent in pred_sent]
            true_toks_list = [wordpunct_tokenize(sent.strip()) for sent in true_sent]
            f1 += sum([word_level_f1(pred_toks, true_toks) for pred_toks, true_toks in zip(pred_toks_list, true_toks_list)]) / len(pred_sent)
            bleu += sum([bleu_metric.compute(predictions=[pred], references=[[true]])['score'] for pred, true in zip(pred_sent, true_sent)]) / len(pred_sent)
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
        output_dir = ('/').join(self.hparams.checkpoint.split('/')[:-3])
        with open(output_dir + '/result.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')


        result = {
            'ppl': avg_ppl,
            'f1': avg_f1,
            'bleu': avg_bleu
        }

        return result


