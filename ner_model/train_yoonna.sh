##!/bin/bash
#echo 'n, y 학습'
#{
#CUDA_VISIBLE_DEVICES=1 python ner_model/train_refiner.py --ner_coef 0.5 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5 > nohup_gen1_ner0.5
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner1_E100_metric > nohup_focus_gen1_ner1_E100.out &
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.3_E100_metric > nohup_focus_gen1_ner0.3_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.5_E100_metric > nohup_focus_gen1_ner0.5_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.1_E100_metric > nohup_focus_gen1_ner0.1_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.7_E100_metric > nohup_focus_gen1_ner0.7_E100.out &
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/wow/gen1_ner0.3_E100_metric > nohup_wow_refine.out &
#CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner1_E100_metric > nohup_wow_gen1_ner1_E100.out &
#CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_metric > nohup_wow_gen1_ner0.3_E100.out &
#CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.5_E100_metric > nohup_wow_gen1_ner0.5_E100.out &
#CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.1_E100_metric > nohup_wow_gen1_ner0.1_E100.out &
#CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.7_E100_metric > nohup_wow_gen1_ner0.7_E100.out &



#yoonna
#ner
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > logs/wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100 --ptuning True > logs/w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init keyword > logs/w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init entity > logs/w_prompt_entity_100.log &

#gen
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100 > logs/gen_wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100 --ptuning True > logs/gen_w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/ --ptuning True --target_word_to_init keyword > logs/gen_w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100 --ptuning True --target_word_to_init entity > logs/gen_w_prompt_entity_100.log &

#fewshot 100
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot --fewshot True --fewnum 100 > logs/gen_wo_prompt_1_100shot.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot --ptuning True --fewshot True --fewnum 100 > logs/gen_w_prompt_rand_1_100shot.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_keyword_1_100shot.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_entity_1_100shot.log &

#fewshot 500
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot --fewshot True --fewnum 100 > logs/gen_wo_prompt_1_100shot.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot --ptuning True --fewshot True --fewnum 100 > logs/gen_w_prompt_rand_1_100shot.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_keyword_1_100shot.log &
#CUDA_VISIBLE_DEVICES=3 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_entity_1_100shot.log &


