#
##ORIGINAL
##########################Ptuning 1 size 10 SHOT###################################
#
##
###########################Ptuning 1 size 50 SHOT###################################
##
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_50shot_a_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data a \
##--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_ori_50shot_a_1.log &&
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_50shot_b_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data b \
##--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_ori_50shot_b_1.log &&
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_50shot_c_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data c \
##--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_ori_50shot_c_1.log &&
##
##
##
###########################Ptuning 1 size 100 SHOT###################################
##

#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 1,1,1 \
#--flag ptuning1_ori_100shot_a_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_ori_100shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 1,1,1 \
#--flag ptuning1_ori_100shot_b_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_ori_100shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 1,1,1 \
#--flag ptuning1_ori_100shot_c_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_ori_100shot_c_1.log &&


#
##########################Ptuning 1 size 500 SHOT###################################
#
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_500shot_a_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data a \
##--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_ori_500shot_a_1.log &&
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_500shot_b_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data b \
##--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_ori_500shot_b_1.log &&
##
##CUDA_VISIBLE_DEVICES=5 python train.py \
##--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
##--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
##--template 1,1,1 \
##--flag ptuning1_ori_500shot_c_1 \
##--ptuning True \
##--seed 644128 \
##--few_shot_setting True \
##--few_shot_data c \
##--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_ori_500shot_c_1.log &&
##
#
##
###REVISED
#
#
##########################Ptuning 1 size 50 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_50shot_a_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_rev_50shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_50shot_b_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_rev_50shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_50shot_c_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 50 > train_ptuning1_fewshot_log/ptuning1_rev_50shot_c_1.log &&
#
#
#
##########################Ptuning 1 size 100 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_100shot_a_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_rev_100shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_100shot_b_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_rev_100shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_100shot_c_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 100 > train_ptuning1_fewshot_log/ptuning1_rev_100shot_c_1.log &&

#
#
##########################Ptuning 1 size 500 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_500shot_a_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data a \
#--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_rev_500shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_500shot_b_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data b \
#--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_rev_500shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_rev_500shot_c_1 \
#--ptuning True \
#--seed 644128 \
#--few_shot_setting True \
#--few_shot_data c \
#--few_shot_num 500 > train_ptuning1_fewshot_log/ptuning1_rev_500shot_c_1.log &&
#
