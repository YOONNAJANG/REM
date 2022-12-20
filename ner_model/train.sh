##!/bin/bash
#echo 'n, y 학습'
#{


##CUDA_VISIBLE_DEVICES=1 python ner_model/train_refiner.py --ner_coef 0.5 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5 > nohup_gen1_ner0.5
##CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner1_E100_metric_weighted > nohup_gen1_ner1_E100_metric_weighted &
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner1_E100_metric > nohup_gen1_ner1_E100_metric &
#C#UDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5_E100_metric_weighted > nohup_gen1_ner0.5_E100_metric_weighted &
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner0.5_E100_metric > nohup_gen1_ner0.5_E100_metric &
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir /home/data/ssh5131/focus_modeling/regen_add_ner/gen1_ner1_E100_metric > nohup_gen1_ner1_E100_metric &

CUDA_VISIBLE_DEVICES=7 python ner_model/train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/gen1_ner0.5_ptuning --ptuning True

#}

