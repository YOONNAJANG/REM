from setproctitle import setproctitle
setproctitle("suhyun")

import os, json
import logging
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from data_utils_refine import add_special_tokens_test, special_tokens_focus, dataloader_focus_test, dataloader_wow_test, add_special_tokens_, dataloader_cmudog_test, dataloader_chatgpt_test
#dataloader_cmudog_test
from datasets import load_metric
import re
from tqdm import tqdm

# from ptuning import get_embedding_layer, PromptEncoder, get_vocab_by_strategy

from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
from metrics.dae_factuality.evaluate_factuality import score_example_single_context
from metrics.distinctN import distinct_n_sentence_level

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()


    def test(self):

        # print("Load DAE model weights")
        # from transformers import ElectraConfig, ElectraTokenizer
        # from metrics.dae_factuality.utils import ElectraDAEModel
        # dae_config_class, dae_model_class, dae_tokenizer_class = ElectraConfig, ElectraDAEModel, ElectraTokenizer
        # dae_tokenizer = dae_tokenizer_class.from_pretrained("/home/data/ssh5131/focus_modeling/model/dae_w_syn_hallu")
        # dae_model = dae_model_class.from_pretrained("/home/data/ssh5131/focus_modeling/model/dae_w_syn_hallu")
        # dae_model.to("cuda")
        #
        # print("Load NER tagger")
        # from flair.data import Sentence
        # from flair.models import SequenceTagger
        # tagger = SequenceTagger.load("flair/ner-english-large")

        r1 = 0
        r2 = 0
        rl = 0
        bleu = 0
        chrf = 0
        dae = 0
        dist1 = 0
        dist2 = 0
        tc = 0
        k_bleu = 0
        ec = 0
        rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bleu_metric = load_metric("sacrebleu")
        chrf_metric = CHRFScore()

        result_list = list()
        chatgpt_data = json.load(open(self.hparams.test_dataset_path))

        for test_data_index, test_data in enumerate(tqdm(chatgpt_data["text_result"])):
            pred_dict = dict()

            # pred_reply = test_data["chatgpt_bad_reaponse"]
            pred_reply = test_data["pred"]
            gold_reply = test_data["gold"]
            knowledge = test_data["knoweldge"]

            # ROUGE

            r = rouge_metric.score(pred_reply[0], gold_reply)
            r1 += r['rouge1'].fmeasure
            r2 += r['rouge2'].fmeasure
            rl += r['rougeL'].fmeasure

            # sacre BLEU
            bleu += bleu_metric.compute(predictions=pred_reply, references=[[gold_reply]])['score']

            # knowledge overlapping (knowledge-BLEU)
            if self.hparams.num_return_sequences > 1:
                for pred_reply_item in pred_reply:
                    k_bleu += bleu_metric.compute(predictions=[pred_reply_item], references=[[knowledge]])['score']
            else:
                k_bleu += bleu_metric.compute(predictions=pred_reply, references=[[knowledge]])['score']
            # ChrF++
            pred_reply_wo_specialchar = re.sub("[^A-Z|\s]", "", pred_reply[0], 0, re.IGNORECASE)
            chrf += chrf_metric([pred_reply_wo_specialchar], [[gold_reply]]).clone().detach()


            # # print('dae')
            # # dae_factuality
            #
            # pred_reply_wo_specialchar = re.sub("[^A-Z|\s]", "", pred_reply[0], 0, re.IGNORECASE)
            # pred_reply_wo_specialchar = pred_reply_wo_specialchar.strip()
            # if len(pred_reply_wo_specialchar) == 0 :
            #     dae += 0
            # else:
            #     dae += score_example_single_context(pred_reply_wo_specialchar, knowledge, dae_model, dae_tokenizer,
            #                                     self.hparams)
            # # print('dae_score', dae)
            #
            # # print('distN')
            # # distinct-N
            #
            # dist1 += distinct_n_sentence_level(pred_reply[0], 1)
            # dist2 += distinct_n_sentence_level(pred_reply[0], 2)
            #
            #
            # # print("TC")
            # pred_format = {'LOC': {"keyword": []},
            #                'MISC': {"keyword": []},
            #                'PER': {"keyword": []},
            #                'ORG': {"keyword": []},
            #                }
            # gold_format = {'LOC': {"keyword": []},
            #                'MISC': {"keyword": []},
            #                'PER': {"keyword": []},
            #                'ORG': {"keyword": []},
            #                }
            # knowledge_format = {'LOC': {"keyword": []},
            #                'MISC': {"keyword": []},
            #                'PER': {"keyword": []},
            #                'ORG': {"keyword": []},
            #                }
            # tmp_ec = 0
            # tmp_tc = 0
            # if pred_reply[0] != "":
            #
            #     #pred_reply, gold_reply
            #     sentence = Sentence(pred_reply[0])
            #     tagger.predict(sentence)
            #     for entity in sentence.get_spans('ner'):
            #         if len(pred_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
            #                 pred_format[entity.get_label("ner").value]["keyword"]:
            #             # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
            #             pred_format[entity.get_label("ner").value]["keyword"].append(entity.text)
            #     sentence = Sentence(gold_reply)
            #     tagger.predict(sentence)
            #     for entity in sentence.get_spans('ner'):
            #         if len(gold_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
            #                 gold_format[entity.get_label("ner").value]["keyword"]:
            #             # format[entity.get_label("ner").value]["keyword"] = format[entity.get_label("ner").value]["keyword"].append(entity.text)
            #             gold_format[entity.get_label("ner").value]["keyword"].append(entity.text)
            #     sentence = Sentence(knowledge)
            #     tagger.predict(sentence)
            #     for entity in sentence.get_spans('ner'):
            #         if len(knowledge_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
            #                 knowledge_format[entity.get_label("ner").value]["keyword"]:
            #             # format[entity.get_label("ner").value]["keyword"] = format[e  ntity.get_label("ner").value]["keyword"].append(entity.text)
            #             knowledge_format[entity.get_label("ner").value]["keyword"].append(entity.text)
            #
            #
            #     for key in gold_format.keys():
            #         gold_w_num = len(gold_format[key]["keyword"])
            #         pred_w_num = len(pred_format[key]["keyword"])
            #         if gold_w_num == 0:
            #             continue
            #         tc_ratio = pred_w_num / gold_w_num
            #         tmp_tc += tc_ratio
            #     tmp_tc = tmp_tc /4
            #
            #
            #     # print("EC")
            #     pred_k_list = []
            #     gold_k_list = []
            #     knowledge_k_list = []
            #     for key in gold_format.keys():
            #         pred_k_list.extend(pred_format[key]["keyword"])
            #         gold_k_list.extend(gold_format[key]["keyword"])
            #         knowledge_k_list.extend(knowledge_format[key]["keyword"])
            #
            #     knowledge_gold = list(set(knowledge_k_list) & set(gold_k_list))
            #     # print(knowledge_gold)
            #     knowledge_gold_pred = list(set(knowledge_gold) & set(pred_k_list))
            #     # print(knowledge_gold_pred)
            #     if len(knowledge_gold) == 0:
            #         tmp_ec = 0
            #     else:
            #         tmp_ec = len(knowledge_gold_pred) / len(knowledge_gold)
            #
            # tc += tmp_tc
            # ec += tmp_ec


        chrf_result = chrf / (test_data_index + 1)
        rouge1_result = r1 / (test_data_index + 1)
        rouge2_result = r2 / (test_data_index + 1)
        rougel_result = rl / (test_data_index + 1)
        bleu_result = bleu / (test_data_index + 1)
        dae_result = dae / (test_data_index + 1)
        dist1_result = dist1 / (test_data_index + 1)
        dist2_result = dist2 / (test_data_index + 1)
        tc_result = tc / (test_data_index + 1)
        ec_result = ec / (test_data_index + 1)
        k_bleu_result = k_bleu / ((test_data_index + 1) * self.hparams.num_return_sequences)


        result_dict = dict()
        result_dict['chrF++'] = chrf_result.item()
        result_dict['rouge1'] = rouge1_result
        result_dict['rouge2'] = rouge2_result
        result_dict['rougeL'] = rougel_result
        result_dict['bleu'] = bleu_result
        result_dict['tc'] = tc_result
        result_dict['ec'] = ec_result
        result_dict['dae_result'] = dae_result
        result_dict['dist1_result'] = dist1_result
        result_dict['dist2_result'] = dist2_result
        result_dict['k_bleu'] = k_bleu_result

        print(result_dict.items())

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list
        print(result_dict)
        with open(self.hparams.output_dir + self.hparams.flag + '.json', 'w')  as outputfile:
            json.dump(result_dict, outputfile, indent='\t')


def main():

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, cmudog}")
    parser.add_argument("--test_dataset_path", type=str, default="/home/mnt/ssh5131/FoCus_data/our_data/test_ours.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str,
                        default='/home/mnt/ssh5131/FoCus_data/our_data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5, LED, transformer-encdec}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-base", help="pretraind_model path") #facebook/bart-base
    parser.add_argument("--mode", type=str, default="gen_imp", help="{ner, gen_exp, gen_imp, original}")
    parser.add_argument("--ckpt", type=str, default="facebook/bart-base", help="ckpt path") #facebook/bart-base
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--random_seed", type=int, default=644128)
    parser.add_argument("--precision", type=int, default=32, help="{16,32,64}")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=512, help="maximum length")
    parser.add_argument("--min_length", type=int, default=32, help="minimum length")
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="{1, 2, 5, 10}, 1 for 1 generated result")
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/eval_output/focus_refiner/", help="default value for PLMs")
    parser.add_argument("--dae_model", type=str, default="/home/data/ssh5131/focus_modeling/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    parser.add_argument("--seed", type=int, default=19981014, help="Seed")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", type=bool, default=True)

    #for p-tuning
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="50,50,50") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])


    args = vars(parser.parse_args())
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])



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

    model.test()
if __name__ == "__main__":
    main()


