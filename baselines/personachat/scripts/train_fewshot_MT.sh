#
#
####Manual prompt models ####
#
##original


#
## 50 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 50 \
#--flag MT_ori_50shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_50shot_a_1.log &&
#

#
## 100 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 100 \
#--flag MT_ori_100shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_100shot_a_1.log &&
#

#
#
## 500 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 500 \
#--flag MT_ori_500shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_500shot_a_1.log &&

#


#
## 50 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 50 \
#--flag MT_ori_50shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_50shot_b_1.log &&
#

#
#
## 100 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 100 \
#--flag MT_ori_100shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_100shot_b_1.log &&
#

#
#
## 500 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 500 \
#--flag MT_ori_500shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_500shot_b_1.log &&
#

#

#
## 50 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 50 \
#--flag MT_ori_50shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_50shot_c_1.log &&
#
#
#
## 100 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 100 \
#--flag MT_ori_100shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_100shot_c_1.log &&
#

#
#
## 500 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 500 \
#--flag MT_ori_500shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_ori_500shot_c_1.log &&
#

#
##revised
#
##50shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 50 \
#--flag MT_rev_50shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_50shot_a_1.log &&
#
#
##100shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 100 \
#--flag MT_rev_100shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_100shot_a_1.log &&

#
##50shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 500 \
#--flag MT_rev_500shot_a_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_500shot_a_1.log &&
#

#


#
##50shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 50 \
#--flag MT_rev_50shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_50shot_b_1.log &&
#

#
##100shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 100 \
#--flag MT_rev_100shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_100shot_b_1.log &&
#

#
##500 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 500 \
#--flag MT_rev_500shot_b_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_500shot_b_1.log &&
#




#
##50shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 50 \
#--flag MT_rev_50shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_50shot_c_1.log &&
#

#
##100shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 100 \
#--flag MT_rev_100shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_100shot_c_1.log &&
#

#
##500 shot
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 500 \
#--flag MT_rev_500shot_c_1 \
#--manual_tuning True > train_fewshot_log/MT_rev_500shot_c_1.log &&
#

