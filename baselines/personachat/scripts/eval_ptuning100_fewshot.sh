
#ORIGINAL

#
#
#
#########################Ptuning 100 size 50 SHOT###################################


CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_50shot_a_1/epoch9-ppl931.7115.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_50shot_a_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_50shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_50shot_b_1/epoch9-ppl919.4457.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_50shot_b_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_50shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_50shot_c_1/epoch9-ppl941.4042.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_50shot_c_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_50shot_c_1.log &&


#
##########################Ptuning 100 size 100 SHOT###################################
#

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_100shot_a_1/epoch9-ppl593.7733.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_100shot_a_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_100shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_100shot_b_1/epoch9-ppl584.3196.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_100shot_b_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_100shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_100shot_c_1/epoch9-ppl612.4029.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_100shot_c_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_100shot_c_1.log &&

#
#
##########################Ptuning 100 size 500 SHOT###################################
#

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_500shot_a_1/epoch9-ppl122.0828.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_500shot_a_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_500shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_500shot_b_1/epoch9-ppl114.5996.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_500shot_b_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_500shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_ori_500shot_c_1/epoch9-ppl113.9540.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 100,100,100 \
--flag ptuning100_ori_500shot_c_1 \
--ptuning True > eval_ptuning100_fewshot_log/ptuning100_ori_500shot_c_1.log


#
#
##REVISED

##########################Ptuning 100 size 50 SHOT###################################

#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_50shot_a_1/epoch9-ppl1188.5537.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_50shot_a_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_50shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_50shot_b_1/epoch9-ppl1076.9438.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_50shot_b_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_50shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_50shot_c_1/epoch9-ppl1076.6538.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_50shot_c_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_50shot_c_1.log &&
#
#
#
###########################Ptuning 100 size 100 SHOT###################################
##
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_100shot_a_1/epoch9-ppl633.2059.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_100shot_a_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_100shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_100shot_b_1/epoch9-ppl632.1283.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_100shot_b_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_100shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_100shot_c_1/epoch9-ppl701.5394.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_100shot_c_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_100shot_c_1.log &&
#
#
##
##
##
###########################Ptuning 100 size 500 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_500shot_a_1/epoch9-ppl142.1937.ckpt/global_step143/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_500shot_a_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_500shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_500shot_b_1/epoch9-ppl145.2836.ckpt/global_step145/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_500shot_b_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_500shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=1 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_rev_500shot_c_1/epoch9-ppl146.9566.ckpt/global_step143/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_rev_500shot_c_1 \
#--ptuning True > eval_ptuning100_fewshot_log/ptuning100_rev_500shot_c_1.log
#
