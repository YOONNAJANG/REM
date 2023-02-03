import os, json
import logging
from argparse import ArgumentParser
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from data_utils_refine import add_special_tokens_test, special_tokens_focus, dataloader_focus_test
from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


logger = logging.getLogger(__file__)




class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.do_sample = self.hparams.do_sample
        self.num_beams = self.hparams.num_beams
        self.top_k = self.hparams.top_k
        self.no_repeat_ngram_size = self.hparams.no_repeat_ngram_size

        from transformers import AutoTokenizer, BartConfig, BartTokenizer
        from refiner_modules import BartEncDec
        from transformers import BartForConditionalGeneration
        self.config = BartConfig.from_pretrained(self.hparams.pretrained_model)
        self.model = BartEncDec.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.congenmodel = BartForConditionalGeneration.from_pretrained(self.hparams.pretrained_model, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        # self.model.to(self.hparams.device)
        self.tokenizer, self.model, self.congenmodel = add_special_tokens_test(self.model, self.congenmodel, self.tokenizer, special_tokens=special_tokens_focus)
        #add_special_tokens_(self.model, self.tokenizer, special_tokens=)
        self.model.to(self.hparams.device)
        self.congenmodel.to(self.hparams.device)
        self.model.eval()
        self.congenmodel.eval()
        if len(self.hparams.checkpoint) > 0:
            checkpoint = torch.load(self.hparams.checkpoint)['state_dict']
            checkpoint = {k[6:]: v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint)
            self.congenmodel.load_state_dict(checkpoint, strict=False)




    def step(self, batch, batch_idx):

        input_ids, decoder_input_ids, lm_labels, ner_labels = batch

        with torch.no_grad():
            output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels, ner_labels=ner_labels)

        return output

    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, lm_labels, ner_labels = batch
        mask = (lm_labels != self.tokenizer.pad_token_id)
        reply = lm_labels[mask]
        results = self.step(batch, batch_idx)
        # print(result.items()) # ner_logits, ner_loss, lm_logits, lm_loss, ner_results
        lm_logits = results["lm_logits"]

        result = {}
        for k, v in results.items():
            if k != "ner_results":
                result[k] = v.detach().cpu()
            else:
                result[k] = v


        lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_criterion(lm_logits_flat_shifted, lm_labels_flat_shifted)
        ppl = torch.exp(lm_loss)

        with torch.no_grad():
            num_beams = self.hparams.num_beams
            # print(input_ids)
            out_ids = self.congenmodel.generate(input_ids=input_ids,do_sample=self.do_sample, num_beams=self.num_beams, top_k=self.top_k, no_repeat_ngram_size=self.no_repeat_ngram_size)
        # print((reply == -100).nonzero(as_tuple=True))
        if len((reply == -100).nonzero(as_tuple=True)[0]) == 0:

            reply = self.tokenizer.decode(reply.tolist(), skip_special_tokens=True)
        else:
            reply_ind = (reply == -100).nonzero(as_tuple=True)[0][0]
            reply = self.tokenizer.decode(reply[:reply_ind].tolist(), skip_special_tokens=True)


        # print(reply)
        # print(input_ids)
        input_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        if num_beams > 1:
            out_ids = [self.tokenizer.decode(output_item, skip_special_tokens=True) for output_item in out_ids.tolist()]
        else:
            out_ids = self.tokenizer.decode(out_ids.squeeze(0).tolist(), skip_special_tokens=True)

        print('input: ', input_text, '\n true: ', reply, '\n pred: ', out_ids)
        result = dict()
        print('ppl: ', ppl)
        result['ppl'] = ppl
        result['y_true_text'] = reply  # tokenize!!!
        result['y_pred_text'] = out_ids
        result['input_text'] = input_text
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
        else:
            text_result = []
            for index, i in enumerate(outputs):
                text_dict = dict()
                text_dict['ppl'] = i['ppl']
                text_dict['y_true_text'] = i['y_true_text']
                text_dict['y_pred_text'] = i['y_pred_text']
                text_dict['input_text'] = i['input_text']
                # text_dict['model_pred_knowledge'] = i['model_pred_knowledge']

                text_result.append(text_dict)

            result = {'text_result': text_result}
            return result

    def test_epoch_end(self, outputs):
        result = self.epoch_end(outputs, state='test')
        print("Load FactCC model weights")
        from transformers import BertTokenizer, BertConfig
        from metrics.factcc import BertPointer
        factcc_config = BertConfig.from_pretrained(self.hparams.factcc_model + '/config.json')
        factcc_model = BertPointer.from_pretrained(self.hparams.factcc_model + '/pytorch_model.bin',
                                                   config=factcc_config)
        factcc_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        factcc_model.to(self.hparams.device)
        factcc_model.eval()

        print("Load DAE model weights")
        from transformers import ElectraConfig, ElectraTokenizer
        from metrics.dae_factuality.utils import ElectraDAEModel
        dae_config_class, dae_model_class, dae_tokenizer_class = ElectraConfig, ElectraDAEModel, ElectraTokenizer
        dae_tokenizer = dae_tokenizer_class.from_pretrained(self.hparams.dae_model)
        dae_model = dae_model_class.from_pretrained(self.hparams.dae_model)
        dae_model.to(self.hparams.device)

        text_result = result['text_result']
        ppl = 0
        r1 = 0
        r2 = 0
        rl = 0
        bleu = 0
        chrf = 0
        factcc_output = 0
        dae = 0
        dist1 = 0
        dist2 = 0
        rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bleu_metric = load_metric("sacrebleu")
        chrf_metric = CHRFScore()

        result_list = list()

        for test_data_index, test_data in enumerate(text_result):
            pred_dict = dict()
            ppl += test_data['ppl']
            print('ppl accumulated: ', ppl)
            gold_reply = test_data['y_true_text']
            pred_reply = test_data['y_pred_text']
            input = test_data['input_text']

            pred_dict['input'] = input
            pred_dict['gold'] = gold_reply
            pred_dict['pred'] = pred_reply

            result_list.append(pred_dict)

            # ROUGE
            if self.hparams.num_beams > 1:
                for pred_reply_item in pred_reply:
                    r = rouge_metric.score(pred_reply_item, gold_reply)
                    r1 += r['rouge1'].fmeasure
                    r2 += r['rouge2'].fmeasure
                    rl += r['rougeL'].fmeasure
            else:
                r = rouge_metric.score(pred_reply, gold_reply)
                r1 += r['rouge1'].fmeasure
                r2 += r['rouge2'].fmeasure
                rl += r['rougeL'].fmeasure

            # sacre BLEU
            if self.hparams.num_beams > 1:
                for pred_reply_item in pred_reply:
                    bleu += bleu_metric.compute(predictions=[pred_reply_item], references=[[gold_reply]])['score']
            else:
                bleu += bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']

            # ChrF++
            if self.hparams.num_beams > 1:
                for pred_reply_item in pred_reply:
                    chrf += chrf_metric([pred_reply_item], [[gold_reply]]).clone().detach()

            else:
                chrf += chrf_metric([pred_reply], [[gold_reply]]).clone().detach()



            # print('factcc')
            # # FactCC
            # if self.hparams.num_beams > 1:
            #     knowledge_input = factcc_tokenizer.tokenize(model_pred_knowledge)
            #     for pred_reply_item in pred_reply:
            #         generated_input = factcc_tokenizer.tokenize(pred_reply_item)
            #         factcc_input = [factcc_tokenizer.cls_token] + knowledge_input + [
            #             factcc_tokenizer.sep_token] + generated_input + [factcc_tokenizer.sep_token]
            #         factcc_input = torch.tensor(factcc_tokenizer.convert_tokens_to_ids(factcc_input)).to(
            #             self.hparams.device).unsqueeze(0)
            #         with torch.no_grad():
            #             factcc_output += factcc_model(factcc_input).argmax().item()
            # else:
            #     knowledge_input = factcc_tokenizer.tokenize(model_pred_knowledge)
            #     generated_input = factcc_tokenizer.tokenize(pred_reply)
            #     factcc_input = [factcc_tokenizer.cls_token] + knowledge_input + [
            #         factcc_tokenizer.sep_token] + generated_input + [factcc_tokenizer.sep_token]
            #     factcc_input = torch.tensor(factcc_tokenizer.convert_tokens_to_ids(factcc_input)).to(
            #         self.hparams.device).unsqueeze(0)
            #     with torch.no_grad():
            #         factcc_output += factcc_model(factcc_input).argmax().item()
            #
            # print('dae')
            # # dae_factuality
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         dae += score_example_single_context(pred_reply_item, model_pred_knowledge, dae_model, dae_tokenizer,
            #                                             self.hparams)
            # else:
            #     dae += score_example_single_context(pred_reply, model_pred_knowledge, dae_model, dae_tokenizer,
            #                                         self.hparams)
            # # print('dae_score', dae)
            #
            # print('distN')
            # # distinct-N
            # if self.hparams.num_beams > 1:
            #     for pred_reply_item in pred_reply:
            #         dist1 += distinct_n_sentence_level(pred_reply_item, 1)
            #         dist2 += distinct_n_sentence_level(pred_reply_item, 2)
            # else:
            #     dist1 += distinct_n_sentence_level(pred_reply, 1)
            #     dist2 += distinct_n_sentence_level(pred_reply, 2)

        chrf_result = chrf / ((test_data_index + 1) * self.hparams.num_beams)
        rouge1_result = r1 / ((test_data_index + 1) * self.hparams.num_beams)
        rouge2_result = r2 / ((test_data_index + 1) * self.hparams.num_beams)
        rougel_result = rl / ((test_data_index + 1) * self.hparams.num_beams)
        bleu_result = bleu / ((test_data_index + 1) * self.hparams.num_beams)
        print("ppl :", ppl)
        print("datalen: ", test_data_index + 1)
        ppl_result = ppl / (test_data_index + 1)
        # factcc_result = factcc_output / ((test_data_index + 1) * self.hparams.num_beams)
        # dae_result = dae / ((test_data_index + 1) * self.hparams.num_beams)
        dist1_result = dist1 / ((test_data_index + 1) * self.hparams.num_beams)
        dist2_result = dist2 / ((test_data_index + 1) * self.hparams.num_beams)

        result_dict = dict()
        result_dict['chrF++'] = chrf_result.item()
        result_dict['rouge1'] = rouge1_result
        result_dict['rouge2'] = rouge2_result
        result_dict['rougeL'] = rougel_result
        result_dict['bleu'] = bleu_result
        result_dict['ppl'] = ppl_result.item()

        # result_dict['factcc_result'] = factcc_result
        # result_dict['dae_result'] = dae_result
        result_dict['dist1_result'] = dist1_result
        result_dict['dist2_result'] = dist2_result

        print(result_dict.items())

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list

        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')

        return self.epoch_end(outputs, state='test')



    def dataloader(self):

        if self.hparams.data_type == "focus":
            test_dataset = dataloader_focus_test(self.hparams, self.tokenizer)
        elif self.hparams.data_type == "wow":
            test_dataset = None
        elif self.hparams.data_type == "persona":
            test_dataset = None, None
        return test_dataset

    def test_dataloader(self):
        test_dataset = self.dataloader()
        print("Valid dataset (Batch, Seq length): {}".format(test_dataset.tensors[0].shape))
        return DataLoader(test_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)

def main():

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, persona}")
    parser.add_argument("--test_dataset_path", type=str, default="/home/mnt/ssh5131/FoCus_data/our_data/test_ours.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str,
                        default='/home/mnt/ssh5131/FoCus_data/our_data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--template", type=str, default="50,50,50")
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    parser.add_argument("--ckpt", type=str, default="facebook/bart-base", help="ckpt path") #facebook/bart-base
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
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
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/eval_output/focus_refiner/", help="default value for PLMs")
    parser.add_argument("--factcc_model", type=str, default="/home/data/ssh5131/focus_modeling/factcc/factcc-checkpoint", help="pre-trained factcc model directory")
    parser.add_argument("--dae_model", type=str, default="/home/data/ssh5131/focus_modeling/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    parser.add_argument("--seed", type=int, default=19981014, help="Seed")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", type=bool, default=True)
    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])

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
        'num_sanity_val_steps': 2,  # None if args['test_mode'] else 0
        'deterministic': False,
        'gpus': args['gpu_num'],
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

