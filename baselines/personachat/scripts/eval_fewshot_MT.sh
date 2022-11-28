

###Manual prompt models ####

#original



# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_50shot_a_1/epoch9-ppl19.5253.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_50shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_50shot_a_1.log &&



# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_100shot_a_1/epoch9-ppl17.6574.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_100shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_100shot_a_1.log &&




# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_500shot_a_1/epoch6-ppl18.4271.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_500shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_500shot_a_1.log &&





# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_50shot_b_1/epoch9-ppl19.3682.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_50shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_50shot_b_1.log &&




# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_100shot_b_1/epoch9-ppl17.5675.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_100shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_100shot_b_1.log &&




# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_500shot_b_1/epoch7-ppl14.1352.ckpt/global_step116/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_500shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_500shot_b_1.log &&





# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_50shot_c_1/epoch9-ppl19.7704.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_50shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_50shot_c_1.log &&



# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_100shot_c_1/epoch9-ppl17.6902.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_100shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_100shot_c_1.log &&




# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_ori_500shot_c_1/epoch6-ppl14.3030.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag MT_ori_500shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_ori_500shot_c_1.log &&



#revised

#50shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_50shot_a_1/epoch9-ppl23.7552.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_50shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_50shot_a_1.log &&


#100shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_100shot_a_1/epoch9-ppl22.5250.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_100shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_100shot_a_1.log &&


#500shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_500shot_a_1/epoch6-ppl16.7967.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_500shot_a_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_500shot_a_1.log &&






#50shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_50shot_b_1/epoch9-ppl23.4593.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_50shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_50shot_b_1.log &&



#100shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_100shot_b_1/epoch9-ppl20.4337.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_100shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_100shot_b_1.log &&



#500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_500shot_b_1/epoch6-ppl16.9852.ckpt/global_step101/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_500shot_b_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_500shot_b_1.log &&






#50shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_50shot_c_1/epoch9-ppl36.2532.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_50shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_50shot_c_1.log &&



#100shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_100shot_c_1/epoch9-ppl21.0662.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_100shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_100shot_c_1.log &&



#500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/MT_rev_500shot_c_1/epoch6-ppl17.0122.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag MT_rev_500shot_c_1 \
--manual_tuning True > eval_fewshot_log/MT_rev_500shot_c_1.log


