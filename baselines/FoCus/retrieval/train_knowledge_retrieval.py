import sys, copy
import os
import json
import random
import logging
import datetime
import argparse
import torch
import torch.nn as nn
import numpy as np
from setproctitle import setproctitle

from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)

from torch.utils.tensorboard import SummaryWriter
from transformers import DPRReader, DPRReaderTokenizer
from transformers import BartTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dpr_for_kr import *

from datasets import load_metric
bert_score_metric = load_metric('bertscore')

import wandb
wandb.login()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set the GPU 2 to use

print("======================================================================")
# logger setting ######################
now = datetime.datetime.now()
save_model_dir_name = now.strftime("model_%Y%m%d_%H%M%S")
save_model_path = f"./models/{save_model_dir_name}/"
os.makedirs(save_model_path)

log_filename = now.strftime("log_%Y%m%d_%H%M%S") + ".log"
total_log_path = save_model_path + log_filename

logging.basicConfig(
    filename=total_log_path,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,  # DEBUG,
)

logger = logging.getLogger(__name__)

tensorboard_dir_name = now.strftime("tb_%Y%m%d_%H%M%S")
writer = SummaryWriter(save_model_path + tensorboard_dir_name)
logger.info(f"save_model_dir_name: {save_model_dir_name}")
########################################

# device ###############################
if torch.cuda.is_available():
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    n_gpu = 0
print("# device: {}".format(device))
########################################

print("\n# prepare bart_tokenizer")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def cacluate_bertscore_with_gt(chosen_knowledge1, groundtruth):
    result = bert_score_metric.compute(predictions=chosen_knowledge1, references=groundtruth, lang='en')
    return result['f1'][0]

def main(args):
    wandb.init(project="train_retrieval_e1_l4_a128", entity="leejasonlee")

    print("\n# prepare model")
    model = DPR_for_KR()
    model = model.to(device)

    focus_train_data = []
    with open("../data/train_ours.json", 'r') as read_file:
        json_data = json.load(read_file)
    for each_dialog in json_data["data"]:
        for u_i, each_turn in enumerate(each_dialog["utterance"]):
            focus_train_data.append({"landmark_link": each_dialog["landmark_link"],
                                     "question": "<human> " + each_turn[f"dialogue{u_i + 1}"][-2],
                                     "golden_knowledge": each_turn["knowledge_candidates"][each_turn["knowledge_answer_index"]]})

    focus_valid_data = []
    with open("../data/valid_ours.json", 'r') as read_file:
        json_data = json.load(read_file)
    for each_dialog in json_data["data"]:
        for u_i, each_turn in enumerate(each_dialog["utterance"]):
            focus_valid_data.append({"landmark_link": each_dialog["landmark_link"],
                                     "question": "<human> " + each_turn[f"dialogue{u_i + 1}"][-2],
                                     "golden_knowledge": each_turn["knowledge_candidates"][each_turn["knowledge_answer_index"]]})

    print("\n# prepare train data")
    train_dataset = KR_Dataset(focus_train_data, "train", max_len=args.max_len, dpr_data_cache=args.dpr_train_data_cache)
####################################################################################
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
####################################################################################
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(train_dataloader): {len(train_dataloader)}")

    print("\n# prepare valid data")
    valid_dataset = KR_Dataset(focus_valid_data, "valid", max_len=args.max_len, dpr_data_cache=args.dpr_valid_data_cache)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"len(valid_dataset): {len(valid_dataset)}")
    print(f"len(valid_dataloader): {len(valid_dataloader)}")

    train_model(model, train_dataloader, valid_dataloader, args)


def train_model(model, train_dataloader, valid_dataloader, args):
    wandb.watch(model, log="all")

    # Prepare optimizer and schedule (linear warmup and decay)
    warmup_steps = 0
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    loss_step = 0
    bestmodel = None
    model.zero_grad()
    for epoch in range(1, args.epochs + 1):
        print("\n***** Running training *****")
        logger.info("\n***** Running training *****")
        print("Epoch: {} / {}".format(epoch, int(args.epochs)))
        logger.info("Epoch: {} / {}\n".format(epoch, int(args.epochs)))

        train_loss = 0
        nb_tr_steps = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="epoch {}".format(str(epoch)))
        for model_input_without_gk, model_input_with_gk, _, _ in pbar:
            outputs = model(model_input_without_gk=model_input_without_gk.view([-1, args.max_len]),
                            model_input_with_gk=model_input_with_gk.view([-1, args.max_len]))

            loss = outputs[0]
            loss.requires_grad_(True)
            train_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            loss_step += 1

            nb_tr_steps += 1

            if (loss_step + 1) % args.gradient_accumulation_steps == 0:
                # clip grad
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 10 == 0 and global_step != 0:
                    print("global step: {}, Training loss: {}".format(global_step, train_loss / nb_tr_steps))
                    logger.info("global step: {}, Training loss: {}".format(global_step, train_loss / nb_tr_steps))
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train_loss", train_loss / nb_tr_steps, global_step)
                    wandb.log({"train_loss_for_sweep": train_loss / nb_tr_steps}, step=global_step)

                    if global_step % 100 == 0 and global_step != 0:
                        model.eval()


                        # ***** Running validation *****
                        print("\n***** Running validation *****")
                        logger.info("***** Running validation *****")
                        print("global_step: {}".format(global_step))
                        logger.info("global_step: {}".format(global_step))

                        valid_loss = 0
                        nb_valid_steps = 0
                        for model_input_without_gk, model_input_with_gk, _, _ in tqdm(valid_dataloader, desc="validation"):
                            with torch.no_grad():
                                outputs = model(model_input_without_gk=model_input_without_gk.view([-1, args.max_len]),
                                                model_input_with_gk=model_input_with_gk.view([-1, args.max_len]))

                            tmp_valid_loss = outputs[0]
                            valid_loss += tmp_valid_loss.item()
                            nb_valid_steps += 1
                        valid_loss = valid_loss / nb_valid_steps
                        print("valid_loss: {}\n".format(valid_loss))
                        logger.info("valid_loss: {}\n".format(valid_loss))
                        writer.add_scalar("valid_loss", valid_loss, global_step)
                        wandb.log({"valid_loss_for_sweep": valid_loss}, step=global_step)

                        # ***** BERTscore *****
                        print("\n***** BERTscore *****")
                        logger.info("***** BERTscore *****")
                        total_bert_score = 0
                        count = 0
                        pbar = tqdm(valid_dataloader, desc="validation")
                        for model_input_without_gk, model_input_with_gk, landmark_link, golden_knowledge in pbar:
                            with torch.no_grad():
                                outputs = model(model_input_without_gk=model_input_without_gk.view([-1, args.max_len]),
                                                model_input_with_gk=model_input_with_gk.view([-1, args.max_len]))

                            bart_dict = total_dic[landmark_link[0]]["bart"]
                            sort_rl = np.argpartition(outputs[2], -1)
                            sort_rl = sort_rl[::-1].tolist()
                            sorted_knowledge = [bart_dict[x] for x in sort_rl[:1]]
                            bart_dict = {}

                            chosen_knowledge1 = sorted_knowledge[0]
                            dec_chosen_knowledge1 = bart_tokenizer.decode(chosen_knowledge1[1:-1])  # <s>, </s> 토큰 제거
                            bert_score = cacluate_bertscore_with_gt([dec_chosen_knowledge1], [golden_knowledge[0]])
                            # if bert_score < 0 or bert_score > 1:
                            #     print("bert_score < 0 or bert_score > 1:", bert_score)
                            total_bert_score += bert_score
                            count += 1
                            pbar.set_postfix({'total_bert_score': total_bert_score / count})

                        print(f"total_bert_score: {total_bert_score / count}\n")
                        logger.info("total_bert_score: {}\n".format(total_bert_score / count))
                        writer.add_scalar("total_bert_score", total_bert_score / count, global_step)
                        wandb.log({"total_bert_score": total_bert_score / count}, step=global_step)


                        model.train()

        # save model
        bestmodel = copy.deepcopy(model)
        save_filename = f"epoch_{epoch}.pt"
        torch.save(bestmodel.state_dict(), save_model_path + save_filename)
        print(f"-> save model: {save_model_path + save_filename}\n")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    setproctitle("focus_train_knowledge_retrieval")

    if "models" not in os.listdir("./"):
        os.makedirs("./models/")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dpr_train_data_cache",
        type=str,
        default=None,
        help='"Path or url of the dataset cache"',
    )
    parser.add_argument(
        "--dpr_valid_data_cache",
        type=str,
        default=None,
        help='"Path or url of the dataset cache"',
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="max_len",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=128,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()

    logger.info("* running file dir name: {}".format(os.path.abspath(__file__)))
    logger.info("* running file name: {}".format(os.path.basename(__file__)))

    logger.info("--epochs={}".format(args.epochs))
    logger.info("--batch_size={}".format(args.batch_size))
    logger.info("--learning_rate={}".format(args.learning_rate))
    logger.info("--gradient_accumulation_steps={}\n".format(args.gradient_accumulation_steps))

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    main(args)