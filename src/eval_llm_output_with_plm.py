import os, json
import logging
from argparse import ArgumentParser

import torch
from itertools import chain

from pytorch_lightning import LightningModule, seed_everything
from datasets import load_metric
import re
from tqdm import tqdm

from datasets import load_metric
from torchmetrics import CHRFScore
from rouge_score import rouge_scorer
from metrics.dae_factuality.evaluate_factuality import score_example_single_context
from metrics.distinctN import distinct_n_sentence_level

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"

logger = logging.getLogger(__file__)
modified = 0

from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-large")


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.do_sample = self.hparams.do_sample
        self.num_beams = self.hparams.num_beams
        self.num_return_sequences = self.hparams.num_return_sequences
        self.top_k = self.hparams.top_k
        self.max_length = self.hparams.max_length
        self.min_length = self.hparams.min_length
        self.no_repeat_ngram_size = self.hparams.no_repeat_ngram_size
        self.id2label = {0:"O", 1:"B", 2:"I"}
        self.metric = load_metric("seqeval")

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="focus", help="{focus, wow, cmudog}")
    parser.add_argument("--test_dataset_path", type=str, default="/home/data/ssh5131/FoCus_data/our_data/test_ours.json")
    parser.add_argument("--threshold_dataset_path", type=str, default="/home/data/ssh5131/FoCus_data/our_data/test_ours.json")
    parser.add_argument("--pretrained_model", type=str, default="facebook/bart-large", help="pre-trained model path among {facebook/bart-base, t5-base, allenai/led-base-16384, facebook/bart-large, t5-large, allenai/led-large-16384}")
    parser.add_argument("--checkpoint", type=str, default="", help="Path of the model checkpoint")
    parser.add_argument("--flag", type=str, default="", help="add description of the output file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--mode", type=str, default="gen_imp", help="{ner, gen_exp, gen_imp, original}")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--random_seed", type=int, default=644128)
    parser.add_argument("--precision", type=int, default=32, help="{16,32,64}")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--cpu_workers", type=int, default=16)
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--before_refine", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="/home/data/ssh5131/focus_modeling/eval_output/focus_refiner/", help="default value for PLMs")
    parser.add_argument("--dae_model", type=str, default="/home/data/yoonna/Refiner/metrics/dae_factuality/model/dae_w_syn_hallu", help="pre-trained dae model directory")
    parser.add_argument("--dependency_type", type=str, default="enhancedDependencies")
    parser.add_argument("--seed", type=int, default=19981014, help="Seed")
    parser.add_argument("--refine_threshold", type=float, default=0.5, help="0<=threshold<=1")

    parser.add_argument("--max_length", type=int, default=512, help="maximum length")
    parser.add_argument("--min_length", type=int, default=32, help="minimum length")
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus filtering (top-p) before sampling, default=1.0")
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="{1, 2, 5, 10}, 1 for 1 generated result")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="no_repeat_ngram_size")



    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    from data_utils_refine import special_tokens_focus
    from transformers import AutoTokenizer
    from transformers import BartConfig as config
    from transformers import BartForConditionalGeneration as congenmodel
    config = config.from_pretrained(args.pretrained_model)

    congenmodel = congenmodel.from_pretrained(args.pretrained_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.__dict__.update(special_tokens_focus)
    congenmodel.resize_token_embeddings(new_num_tokens=len(tokenizer))

    if len(args.checkpoint) > 0:
        checkpoint = torch.load(args.checkpoint)['state_dict']
        checkpoint_congen = {k[6:]: v for k, v in checkpoint.items()}
        congenmodel.load_state_dict(checkpoint_congen, strict=False)
    congenmodel.to('cuda')

    rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    from transformers import ElectraTokenizer
    from metrics.dae_factuality.utils import ElectraDAEModel
    dae_model_class, dae_tokenizer_class = ElectraDAEModel, ElectraTokenizer
    dae_tokenizer = dae_tokenizer_class.from_pretrained(args.dae_model)
    dae_model = dae_model_class.from_pretrained(args.dae_model)
    dae_model.to(args.device)
    test_data_index = 0
    r1 = 0
    r2 = 0
    rl = 0
    bleu = 0
    chrf = 0
    dae = 0
    dist1 = 0
    dist2 = 0
    tc = 0
    ec = 0
    k_bleu = 0
    bleu_metric = load_metric("sacrebleu")
    chrf_metric = CHRFScore()

    result_list = list()

    if args.threshold_dataset_path != "":
        refined_index = []

    else:
        with open(args.threshold_dataset_path, "r", encoding="utf-8") as t_f:
            threshold_dataset = json.loads(t_f.read())['text_result']
            refined_index = []
            for t_i, item in enumerate(threshold_dataset):
                if item['refine'] == 'True':
                    refined_index.append(t_i)

    original_focus_path = "/home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json"
    if args.data_type == "focus":
        with open(original_focus_path, "r", encoding="utf-8") as focus_f:
            original_focus = json.loads(focus_f.read())

    with open(args.test_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
        for dial_idx, dialogue in enumerate(tqdm(dataset)):
            utterance = dialogue["utterance"]
            new_dialogue = dict()
            new_dialogue["utterance"] = list()
            for i, utt in enumerate(utterance):
                key = "dialogue" + str(i + 1)
                if key not in utt.keys():
                    continue

                if args.data_type == "focus" and 'llm_gen_to_llm' not in args.test_dataset_path:
                    knowledge = utt['knowledge_candidates'][utt['knowledge_answer_index']]
                else:
                    knowledge = utt["selected_knowledge"]

                original_output = utt['output']

                original_output_dae = score_example_single_context(original_output, knowledge, dae_model, dae_tokenizer, args)
                if original_output_dae < args.refine_threshold:
                    refined_index.append(test_data_index)

                if args.before_refine == True:
                    pred_reply = utt["output"]
                else:
                    if test_data_index not in refined_index:
                        pred_reply = utt["output"]
                    else: #BART generates new results
                        if args.data_type == "focus":
                            bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
                            human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
                            persona_st = tokenizer.convert_tokens_to_ids(tokenizer.persona_token)
                            knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)
                            persona = original_focus[dial_idx]['persona']
                            persona_grounding = original_focus[dial_idx]['utterance'][i]['persona_grounding']
                            selected_persona = [p for p,g in zip(persona,persona_grounding) if g==True]
                            if len(selected_persona) > 0:
                                selected_persona = [[persona_st] + tokenizer.encode(p)[1:-1] for p in selected_persona]
                            selected_persona = list(chain(*selected_persona))
                            knowledge_enc = tokenizer.encode(knowledge)[1:-1][:128]
                            prev_utt = tokenizer.encode(utt['output'])[1:-1][:128]
                            enc_sequence = [bos] + [knowledge_st] + knowledge_enc + selected_persona + [human_st] + prev_utt + [eos]
                            enc_sequence = torch.LongTensor(enc_sequence).to('cuda').unsqueeze(0)

                            with torch.no_grad():
                                output_ids = congenmodel.generate(input_ids=enc_sequence,
                                                                    do_sample=args.do_sample, num_beams=args.num_beams,
                                                                    num_return_sequences=args.num_return_sequences,
                                                                    top_k=args.top_k,
                                                                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                    min_length=args.min_length,
                                                                    max_length=args.max_length)
                            pred_reply = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


                        else:
                            bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
                            human_st = tokenizer.convert_tokens_to_ids(tokenizer.human_token)
                            knowledge_st = tokenizer.convert_tokens_to_ids(tokenizer.knowledge_token)
                            knowledge_enc = tokenizer.encode(knowledge)[1:-1][:128]
                            prev_utt = tokenizer.encode(utt['output'])[1:-1][:128]
                            enc_sequence = [bos] + [knowledge_st] + knowledge_enc + [human_st] + prev_utt + [eos]
                            enc_sequence = torch.LongTensor(enc_sequence).to('cuda').unsqueeze(0)
                            with torch.no_grad():
                                output_ids = congenmodel.generate(input_ids=enc_sequence,
                                                                    do_sample=args.do_sample, num_beams=args.num_beams,
                                                                    num_return_sequences=args.num_return_sequences,
                                                                    top_k=args.top_k,
                                                                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                                    min_length=args.min_length,
                                                                    max_length=args.max_length)
                            pred_reply = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                gold_reply = utt["labels"]

                # ROUGE
                r = rouge_metric.score(pred_reply, gold_reply)
                r1 += r['rouge1'].fmeasure
                r2 += r['rouge2'].fmeasure
                rl += r['rougeL'].fmeasure

                # sacre BLEU
                bleu += bleu_metric.compute(predictions=[pred_reply], references=[[gold_reply]])['score']

                # knowledge overlapping (knowledge-BLEU)
                k_bleu += bleu_metric.compute(predictions=[pred_reply], references=[[knowledge]])['score']

                # ChrF++
                pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply, 0, re.IGNORECASE)
                chrf += chrf_metric([pred_reply_wo_specialchar], [[gold_reply]])

                # dae_factuality
                pred_reply_wo_specialchar = re.sub("[^\w|\s]", "", pred_reply, 0, re.IGNORECASE)
                pred_reply_wo_specialchar = pred_reply_wo_specialchar.strip()
                knowledge_wo_specialchar = re.sub("[^\w|\s}]", "", knowledge, 0, re.IGNORECASE)
                knowledge_wo_specialchar = knowledge_wo_specialchar.strip()
                if len(pred_reply_wo_specialchar) == 0:
                    dae += 0
                else:
                    dae += score_example_single_context(pred_reply_wo_specialchar, knowledge, dae_model,
                                                        dae_tokenizer, args)

                # distinct-N
                dist1 += distinct_n_sentence_level(pred_reply, 1)
                dist2 += distinct_n_sentence_level(pred_reply, 2)

                pred_format = {'LOC': {"keyword": []},
                               'MISC': {"keyword": []},
                               'PER': {"keyword": []},
                               'ORG': {"keyword": []},
                               }
                gold_format = {'LOC': {"keyword": []},
                               'MISC': {"keyword": []},
                               'PER': {"keyword": []},
                               'ORG': {"keyword": []},
                               }
                knowledge_format = {'LOC': {"keyword": []},
                                    'MISC': {"keyword": []},
                                    'PER': {"keyword": []},
                                    'ORG': {"keyword": []},
                                    }
                tmp_ec = 0
                tmp_tc = 0
                if pred_reply != "":
                    # pred_reply, gold_reply
                    sentence = pred_reply
                    sentence = re.sub("[^\w|\s]", "", sentence, 0, re.IGNORECASE)

                    sentence = Sentence(sentence)
                    tagger.predict(sentence)
                    for entity in sentence.get_spans('ner'):
                        if len(pred_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                                pred_format[entity.get_label("ner").value]["keyword"]:
                            pred_format[entity.get_label("ner").value]["keyword"].append(entity.text)
                    sentence = Sentence(gold_reply)
                    tagger.predict(sentence)
                    for entity in sentence.get_spans('ner'):
                        if len(gold_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                                gold_format[entity.get_label("ner").value]["keyword"]:
                            gold_format[entity.get_label("ner").value]["keyword"].append(entity.text)
                    sentence = Sentence(knowledge)
                    tagger.predict(sentence)
                    for entity in sentence.get_spans('ner'):
                        if len(knowledge_format[entity.get_label("ner").value]["keyword"]) == 0 or entity.text not in \
                                knowledge_format[entity.get_label("ner").value]["keyword"]:
                            knowledge_format[entity.get_label("ner").value]["keyword"].append(entity.text)

                    for key in gold_format.keys():
                        gold_w_num = len(gold_format[key]["keyword"])
                        pred_w_num = len(pred_format[key]["keyword"])
                        if gold_w_num == 0:
                            continue
                        tc_ratio = pred_w_num / gold_w_num
                        tmp_tc += tc_ratio
                    tmp_tc = tmp_tc / 4

                    pred_k_list = []
                    gold_k_list = []
                    knowledge_k_list = []
                    for key in gold_format.keys():
                        pred_k_list.extend(pred_format[key]["keyword"])
                        gold_k_list.extend(gold_format[key]["keyword"])
                        knowledge_k_list.extend(knowledge_format[key]["keyword"])

                    knowledge_gold = list(set(knowledge_k_list) & set(gold_k_list))
                    knowledge_gold_pred = list(set(knowledge_gold) & set(pred_k_list))
                    if len(knowledge_gold) == 0:
                        tmp_ec = 0
                    else:
                        tmp_ec = len(knowledge_gold_pred) / len(knowledge_gold)

                tc += tmp_tc
                ec += tmp_ec
                test_data_index += 1


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
        k_bleu_result = k_bleu / (test_data_index + 1)

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
        result_dict['refined_num'] = len(refined_index)
        result_dict['total_sample_num'] = test_data_index + 1

        test_result = dict()
        for key, value in result_dict.items():
            test_result[key] = value

        result_dict['text_result'] = result_list

        with open(args.output_dir + args.flag + '.json', 'w') as outputfile:
            json.dump(result_dict, outputfile, indent='\t')

if __name__ == "__main__":
    main()
