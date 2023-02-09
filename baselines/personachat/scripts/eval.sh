##Baseline Persona Text
#CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/none_ori_none_ori/epoch6-ppl13.6593.ckpt/global_step898/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_none_original.txt > eval_log/none_ori_none_ori_none_ori.log &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/self_ori_self_ori/epoch6-ppl10.5446.ckpt/global_step898/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_self_original.txt > eval_log/self_ori_self_ori_self_ori.log &
#
#CUDA_VISIBLE_DEVICES=6 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/other_ori_other_ori/epoch7-ppl13.6552.ckpt/global_step1027/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_other_original.txt > eval_log/other_ori_other_ori_other_ori.log &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/both_ori_both_ori/epoch8-ppl10.7132.ckpt/global_step1155/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt > eval_log/both_ori_both_ori_both_ori.log &
#
#CUDA_VISIBLE_DEVICES=4 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/self_rev_self_rev/epoch7-ppl12.8088.ckpt/global_step1027/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_self_revised.txt > eval_log/self_rev_self_rev_self_rev.log &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/other_rev_other_rev/epoch8-ppl13.7107.ckpt/global_step1155/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_other_revised.txt > eval_log/other_rev_other_rev_other_rev.log &
#
#CUDA_VISIBLE_DEVICES=6 nohup python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/both_rev_both_rev/epoch5-ppl12.7930.ckpt/global_step770/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt > eval_log/both_rev_both_rev_both_rev.log &


#########################Ptuning-FULLSHOT###################################

#####Ptuning FULLSHOT with original
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning3_both_ori_both_ori/epoch9-ppl174.1031.ckpt/global_step2567/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 3,3,3 \
#--flag ptuning3_both_ori_both_ori \
#--ptuning True > eval_log/ptuning3_both_ori_both_ori.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_both_ori_both_ori/epoch9-ppl316.1451.ckpt/global_step2567/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 1,1,1 \
#--flag ptuning1_both_ori_both_ori \
#--ptuning True > eval_log/ptuning1_both_ori_both_ori.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_both_ori_both_ori/epoch9-ppl69.8860.ckpt/global_step2567/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 10,10,10 \
#--flag ptuning10_both_ori_both_ori \
#--ptuning True > eval_log/ptuning10_both_ori_both_ori.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_both_ori_both_ori/epoch9-ppl18.9584.ckpt/global_step2567/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_both_ori_both_ori \
#--ptuning True > eval_log/ptuning50_both_ori_both_ori.log &&
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_both_ori_both_ori/epoch6-ppl19.9836.ckpt/global_step1797/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--template 100,100,100 \
#--flag ptuning100_both_ori_both_ori \
#--ptuning True > eval_log/ptuning100_both_ori_both_ori.log &&
#
#
#CUDA_VISIBLE_DEVICES=0 python eval.py \
#--checkpoint /home/mnt/yoonna/personachat/model/bart_base/manual_tuning_both_ori_both_ori/epoch5-ppl10.6461.ckpt/global_step1540/mp_rank_00_model_states.pt \
#--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_original.txt \
#--flag manual_tuning_both_ori_both_ori \
#--manual_tuning True > eval_log/manual_both_ori_both_ori.log









#####Ptuning FULLSHOT with revised data


CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning3_both_rev_both_rev/epoch9-ppl197.4596.ckpt/global_step2567/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 3,3,3 \
--flag ptuning3_both_rev_both_rev \
--ptuning True > eval_log/ptuning3_both_rev_both_rev.log &&


CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning1_both_rev_both_rev/epoch8-ppl360.1115.ckpt/global_step2310/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 1,1,1 \
--flag ptuning1_both_rev_both_rev \
--ptuning True > eval_log/ptuning1_both_rev_both_rev.log &&


CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning10_both_rev_both_rev/epoch8-ppl92.3971.ckpt/global_step2310/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 10,10,10 \
--flag ptuning10_both_rev_both_rev \
--ptuning True > eval_log/ptuning10_both_rev_both_rev.log &&

CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning50_both_rev_both_rev/epoch9-ppl28.3732.ckpt/global_step2567/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 50,50,50 \
--flag ptuning50_both_rev_both_rev \
--ptuning True > eval_log/ptuning50_both_rev_both_rev.log &&

CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/ptuning100_both_rev_both_rev/epoch9-ppl22.4117.ckpt/global_step2567/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--template 100,100,100 \
--flag ptuning100_both_rev_both_rev \
--ptuning True > eval_log/ptuning100_both_rev_both_rev.log &&

CUDA_VISIBLE_DEVICES=7 python eval.py \
--checkpoint /home/mnt/yoonna/personachat/model/bart_base/manual_tuning_both_rev_both_rev/epoch5-ppl12.8942.ckpt/global_step1540/mp_rank_00_model_states.pt \
--test_dataset_path /home/yoonna/persona_chat/data/personachat/test_both_revised.txt \
--flag manual_tuning_both_rev_both_rev \
--manual_tuning True > eval_log/manual_both_rev_both_rev.log

