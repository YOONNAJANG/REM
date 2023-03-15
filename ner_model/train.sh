##!/bin/bash
#echo 'n, y 학습'
# {
#CUDA_VISIBLE_DEVICES=1 python ner_model/train_refiner.py --ner_coef 0.5 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5 > nohup_gen1_ner0.5
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner1_E100_metric > nohup_focus_gen1_ner1_E100.out &
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.3_E100_metric > nohup_focus_gen1_ner0.3_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.5_E100_metric > nohup_focus_gen1_ner0.5_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.1_E100_metric > nohup_focus_gen1_ner0.1_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.7_E100_metric > nohup_focus_gen1_ner0.7_E100.out &

CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow_v3/gen1_ner1_E100_nops > train_log/nohup_wow_v3_gen1_ner1_E100.out &
CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow_v3/gen1_ner0.3_E100_nops > train_log/nohup_wow_v3_gen1_ner0.3_E100.out &
CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow_v3/gen1_ner0.5_E100_nops > train_log/nohup_wow_v3_gen1_ner0.5_E100.out &
CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow_v3/gen1_ner0.1_E100_nops > train_log/nohup_wow_v3_gen1_ner0.1_E100.out &
CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow_v3/gen1_ner0.7_E100_nops > train_log/nohup_wow_v3_gen1_ner0.7_E100.out &


#
##CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type cmudog --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner1_E100 > train_log/nohup_cmudog_gen1_ner1_E100.out &
##CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type cmudog --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100 > train_log/nohup_cmudog_gen1_ner0.3_E100.out &
##CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type cmudog --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.5_E100 > train_log/nohup_cmudog_gen1_ner0.5_E100.out &
##CUDA_VISIBLE_DEVICES=0 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type cmudog --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.1_E100 > train_log/nohup_cmudog_gen1_ner0.1_E100.out &
# CUDA_VISIBLE_DEVICES=5 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type cmudog --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.7_E100 > train_log/nohup_cmudog_gen1_ner0.7_E100.out &


#}
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > logs/wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100 --ptuning True > logs/w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init keyword > logs/w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init entity > logs/w_prompt_entity_100.log &


#CUDA_VISIBLE_DEVICES=3 python ner_model/train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --checkpoint /home/data/yoonna/Refiner/output/regen_add_ner/original_100/epoch27-valid_lm_loss1.4121.ckpt --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/test
#CUDA_VISIBLE_DEVICES=7 python ner_model/train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/gen1_ner0.5_ptuning --ptuning True
#CUDA_VISIBLE_DEVICES=7 python ner_model/train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/gen1_ner0.5_ptuning_fewshot --ptuning True --fewshot True --fewnum 100


# ner0
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0 --epochs 100 --data_type focus --output_dir ./output/ner0_focus > ./output/ner0_focus/nohup_ner0_focus.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0 --epochs 100 --data_type wow --output_dir ./output/ner0_wow > ./output/ner0_wow/nohup_ner0_wow.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0 --epochs 100 --data_type cmudog --output_dir ./output/ner0_cmudog > ./output/ner0_cmudog/nohup_ner0_cmudog.log &