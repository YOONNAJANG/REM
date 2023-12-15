##BART base ner focus
CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.3 --epochs 100 --lr 6.25e-5 --data_type focus --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > /home/data/yoonna/Refiner/ner_model/logs/wo_prompt_100.log &
#BART base ner wow
#CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.7 --epochs 100 --lr 6.25e-5 --data_type wow --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100 > logs/wow_wo_prompt_100.log &
##BART base ner cmudog
#CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --lr 6.25e-5 --data_type cmudog --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100 > logs/cmudog_wo_prompt_100.log &
#
###BART large ner focus
#CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.3 --epochs 100 --pretrained_model facebook/bart-large --data_type focus --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100 > /home/data/yoonna/Refiner/src/logs/bart_large_wo_prompt_100.log &
###BART large ner wow
#CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.7 --epochs 100 --pretrained_model facebook/bart-large --data_type wow --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100 > logs/bart_large_wow_wo_prompt_100.log &
###BART large ner cmudog
#CUDA_VISIBLE_DEVICES=0 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model facebook/bart-large --data_type cmudog --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100 > logs/bart_large_cmudog_wo_prompt_100.log &

python src/train_refiner.py --ner_coef 1 --epochs 100 --data_type focus --output_dir ckpt \
--train_path "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/train.json" \
--train_cache_path "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/our_train_cache.tar.gz" \
--dev_path "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/dev.json" \
--dev_cache_path "/home/data/ssh5131/focus_modeling/for_refiner_v2/focus/our_dev_cache.tar.gz"



python src/train_refiner.py --ner_coef 1 --epochs 100 --data_type wow --output_dir ckpt_wow \
--train_path "/home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/train_beam5_09k_B_B_with_ner.json" \
--train_cache_path "/home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/train_beam5_09k_B_B_with_ner_cache.tar.gz" \
--dev_path "/home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/valid_random_split_beam5_09k_B_B_with_ner.json" \
--dev_cache_path "/home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/valid_random_split_beam5_09k_B_B_with_ner_cache.tar.gz"
