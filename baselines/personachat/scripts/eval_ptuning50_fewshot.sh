
#ORIGINAL

#
##########################Ptuning 50 size 50 SHOT###################################
#
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_50shot_a_1/epoch9-ppl756.4012.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_50shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_50shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_50shot_b_1/epoch9-ppl710.7040.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_50shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_50shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_50shot_c_1/epoch9-ppl894.0371.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_50shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_50shot_c_1.log &&
#
#
#
##########################Ptuning 50 size 100 SHOT###################################
#
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_100shot_a_1/epoch9-ppl496.0901.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_100shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_100shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_100shot_b_1/epoch9-ppl464.3008.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_100shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_100shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_100shot_c_1/epoch9-ppl487.0777.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_100shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_100shot_c_1.log &&
#
#
#
##########################Ptuning 50 size 500 SHOT###################################
#
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_500shot_a_1/epoch9-ppl109.7100.ckpt/global_step143/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_500shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_500shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_500shot_b_1/epoch9-ppl89.6177.ckpt/global_step145/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_500shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_500shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_ori_500shot_c_1/epoch9-ppl134.8366.ckpt/global_step143/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_ori_500shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_ori_500shot_c_1.log &&
#
#
##REVISED
#
##########################Ptuning 50 size 50 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_50shot_a_1/epoch9-ppl915.1452.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_50shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_50shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_50shot_b_1/epoch9-ppl940.4592.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_50shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_50shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_50shot_c_1/epoch9-ppl913.7882.ckpt/global_step15/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_50shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_50shot_c_1.log &&
#
#
##########################Ptuning 50 size 100 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_100shot_a_1/epoch9-ppl523.3843.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_100shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_100shot_a_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_100shot_b_1/epoch9-ppl564.9120.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_100shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_100shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_100shot_c_1/epoch9-ppl555.6290.ckpt/global_step28/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_100shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_100shot_c_1.log &&
#
#
#
##########################Ptuning 50 size 500 SHOT###################################
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_500shot_a_1/epoch9-ppl119.9857.ckpt/global_step143/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_500shot_a_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_500shot_a_1.log &&

#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_500shot_b_1/epoch9-ppl119.0819.ckpt/global_step145/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_500shot_b_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_500shot_b_1.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_rev_500shot_c_1/epoch9-ppl44.4054.ckpt/global_step290/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_rev_500shot_c_1 \
#--ptuning True > eval_ptuning50_fewshot_log/ptuning50_rev_500shot_c_1.log

