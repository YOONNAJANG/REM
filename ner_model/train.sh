##!/bin/bash
#echo 'n, y 학습'
#{
#CUDA_VISIBLE_DEVICES=1 python ner_model/train_refiner.py --ner_coef 0.5 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5 > nohup_gen1_ner0.5
CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner1_E100_metric > nohup_focus_gen1_ner1_E100.out &
CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.3_E100_metric > nohup_focus_gen1_ner0.3_E100.out &
CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.5_E100_metric > nohup_focus_gen1_ner0.5_E100.out &
CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.1_E100_metric > nohup_focus_gen1_ner0.1_E100.out &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.7_E100_metric > nohup_focus_gen1_ner0.7_E100.out &

#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/wow/gen1_ner0.3_E100_metric > nohup_wow_refine.out &
CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner1_E100_metric > nohup_wow_gen1_ner1_E100.out &
CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 0.3 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_metric > nohup_wow_gen1_ner0.3_E100.out &
#CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.5_E100_metric > nohup_wow_gen1_ner0.5_E100.out &
#CUDA_VISIBLE_DEVICES=3 nohup python ner_model/train_refiner.py --ner_coef 0.1 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.1_E100_metric > nohup_wow_gen1_ner0.1_E100.out &
#CUDA_VISIBLE_DEVICES=4 nohup python ner_model/train_refiner.py --ner_coef 0.7 --epochs 100 --data_type wow --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.7_E100_metric > nohup_wow_gen1_ner0.7_E100.out &

#}

