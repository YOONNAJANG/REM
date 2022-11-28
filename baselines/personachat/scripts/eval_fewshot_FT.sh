
########################Ptuning-FULLSHOT###################################

#original


# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_50shot_a_1/epoch9-ppl19.5638.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_50shot_a_1 > eval_fewshot_log/FT_ori_50shot_a_1.log &&




# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_100shot_a_1/epoch9-ppl17.6108.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_100shot_a_1 > eval_fewshot_log/FT_ori_100shot_a_1.log &&



# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_500shot_a_1/epoch6-ppl14.3942.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_500shot_a_1 > eval_fewshot_log/FT_ori_500shot_a_1.log &&




# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_50shot_b_1/epoch9-ppl18.5507.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_50shot_b_1 > eval_fewshot_log/FT_ori_50shot_b_1.log &&




# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_100shot_b_1/epoch9-ppl17.5639.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_100shot_b_1 > eval_fewshot_log/FT_ori_100shot_b_1.log &&


# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_500shot_b_1/epoch7-ppl14.1388.ckpt/global_step116/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_500shot_b_1 > eval_fewshot_log/FT_ori_500shot_b_1.log &&





# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_50shot_c_1/epoch9-ppl18.9295.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_50shot_c_1 > eval_fewshot_log/FT_ori_50shot_c_1.log &&


# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_100shot_c_1/epoch6-ppl17.9961.ckpt/global_step20/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_100shot_c_1 > eval_fewshot_log/FT_ori_100shot_c_1.log &&


# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_ori_500shot_c_1/epoch6-ppl14.3120.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
--flag FT_ori_500shot_c_1 > eval_fewshot_log/FT_ori_500shot_c_1.log &&



#revised



# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_50shot_a_1/epoch9-ppl23.0614.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_50shot_a_1 > eval_fewshot_log/FT_rev_50shot_a_1.log &&


# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_100shot_a_1/epoch9-ppl21.8606.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_100shot_a_1 > eval_fewshot_log/FT_rev_100shot_a_1.log &&


# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_500shot_a_1/epoch6-ppl16.9671.ckpt/global_step100/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_500shot_a_1 > eval_fewshot_log/FT_rev_500shot_a_1.log &&



# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_50shot_b_1/epoch9-ppl22.5634.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_50shot_b_1 > eval_fewshot_log/FT_rev_50shot_b_1.log &&



# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_100shot_b_1/epoch9-ppl20.4973.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_100shot_b_1 > eval_fewshot_log/FT_rev_100shot_b_1.log &&



# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_500shot_b_1/epoch5-ppl16.9021.ckpt/global_step87/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_500shot_b_1 > eval_fewshot_log/FT_rev_500shot_b_1.log &&



# 50 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_50shot_c_1/epoch9-ppl22.8621.ckpt/global_step15/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_50shot_c_1 > eval_fewshot_log/FT_rev_50shot_c_1.log &&


# 100 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_100shot_c_1/epoch9-ppl20.8680.ckpt/global_step28/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_100shot_c_1 > eval_fewshot_log/FT_rev_100shot_c_1.log &&



# 500 shot
CUDA_VISIBLE_DEVICES=0 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/FT_rev_500shot_c_1/epoch5-ppl16.8260.ckpt/global_step86/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag FT_rev_500shot_c_1 > eval_fewshot_log/FT_rev_500shot_c_1.log




