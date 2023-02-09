#
##ORIGINAL
##########################Ptuning 10 size 10 SHOT###################################
#
#
#
#
##########################Ptuning 10 size 50 SHOT###################################
#

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_50shot_a_1/epoch9-ppl592.3036.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_50shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_50shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_50shot_b_1/epoch9-ppl552.2362.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_50shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_50shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_50shot_c_1/epoch9-ppl579.0855.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_50shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_50shot_c_1.log &&

#
#
##########################Ptuning 10 size 100 SHOT###################################
#
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_100shot_a_1/epoch9-ppl440.6215.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_100shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_100shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_100shot_b_1/epoch9-ppl442.9419.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_100shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_100shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_100shot_c_1/epoch9-ppl416.9060.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_100shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_100shot_c_1.log &&
#
#
#
##########################Ptuning 10 size 500 SHOT###################################


CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_500shot_a_1/epoch9-ppl302.3159.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_500shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_500shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_500shot_b_1/epoch9-ppl280.5046.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_500shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_500shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_ori_500shot_c_1/epoch9-ppl302.4047.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--template 10,10,10 \
--flag ptuning10_ori_500shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_ori_500shot_c_1.log &&

#
#
##REVISED

##########################Ptuning 10 size 50 SHOT###################################

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_50shot_a_1/epoch9-ppl669.7402.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_50shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_50shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_50shot_b_1/epoch9-ppl691.5956.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_50shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_50shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_50shot_c_1/epoch9-ppl660.8466.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_50shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_50shot_c_1.log &&
#
##########################Ptuning 10 size 100 SHOT###################################
#
CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_100shot_a_1/epoch9-ppl502.6300.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_100shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_100shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_100shot_b_1/epoch9-ppl477.0415.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_100shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_100shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_100shot_c_1/epoch9-ppl502.7074.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_100shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_100shot_c_1.log &&

#
#
##########################Ptuning 10 size 500 SHOT###################################

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_500shot_a_1/epoch9-ppl308.3965.ckpt/global_step143/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_500shot_a_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_500shot_a_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_500shot_b_1/epoch9-ppl372.6535.ckpt/global_step145/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_500shot_b_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_500shot_b_1.log &&

CUDA_VISIBLE_DEVICES=1 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_rev_500shot_c_1/epoch9-ppl484.2484.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_rev_500shot_c_1 \
--ptuning True > eval_ptuning10_fewshot_log/ptuning10_rev_500shot_c_1.log

