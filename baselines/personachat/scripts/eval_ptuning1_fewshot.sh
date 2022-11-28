#
##ORIGINAL
###########################Ptuning 1 size 50 SHOT###################################
##
##
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_50shot_a_1/epoch9-ppl894.6132.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_50shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_50shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_50shot_b_1/epoch9-ppl852.6581.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_50shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_50shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_50shot_c_1/epoch9-ppl771.6543.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_50shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_50shot_c_1.log &&


##
###########################Ptuning 1 size 100 SHOT###################################
##

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_100shot_a_1/epoch9-ppl698.5541.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_100shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_100shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_100shot_b_1/epoch9-ppl674.7011.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_100shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_100shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_100shot_c_1/epoch9-ppl716.8381.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_100shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_100shot_c_1.log &&


#
##########################Ptuning 1 size 500 SHOT###################################


CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_500shot_a_1/epoch9-ppl408.9417.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_500shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_500shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_500shot_b_1/epoch9-ppl384.5774.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_500shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_500shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_ori_500shot_c_1/epoch9-ppl375.4704.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 1,1,1 \
--flag ptuning1_ori_500shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_ori_500shot_c_1.log &&

#
##
###REVISED
#
#
##########################Ptuning 1 size 50 SHOT###################################
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_50shot_a_1/epoch9-ppl1179.2881.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_50shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_50shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_50shot_b_1/epoch9-ppl967.5659.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_50shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_50shot_b_1.log &&
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_50shot_c_1/epoch9-ppl1101.8674.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_50shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_50shot_c_1.log &&


#
##########################Ptuning 1 size 100 SHOT###################################
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_100shot_a_1/epoch9-ppl800.9291.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_100shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_100shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_100shot_b_1/epoch9-ppl830.9764.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_100shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_100shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_100shot_c_1/epoch9-ppl732.7039.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_100shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_100shot_c_1.log &&

#
#
##########################Ptuning 1 size 500 SHOT###################################
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_500shot_a_1/epoch9-ppl420.7701.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_500shot_a_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_500shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_500shot_b_1/epoch9-ppl439.9572.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_500shot_b_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_500shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_rev_500shot_b_1/epoch9-ppl439.9572.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_rev_500shot_c_1 \
--ptuning True > eval_ptuning1_fewshot_log/ptuning1_rev_500shot_c_1.log

