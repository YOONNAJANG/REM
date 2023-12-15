#bart-large prediction

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_wow_bart-large_refine > eval_logs/threshold_0.0_wow_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_wow_bart-large_refine > eval_logs/threshold_0.1_wow_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_wow_bart-large_refine > eval_logs/threshold_0.2_wow_bart-large_refine.out &

#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_wow_bart-large_refine > eval_logs/threshold_0.3_wow_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_wow_bart-large_refine > eval_logs/threshold_0.4_wow_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_bart-large_refine > eval_logs/threshold_0.5_wow_bart-large_refine.out &

CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_wow_bart-large_refine_only.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_wow_bart-large_refine > eval_logs/threshold_0.6_wow_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_wow_bart-large_refine > eval_logs/threshold_0.7_wow_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_wow_bart-large_refine > eval_logs/threshold_0.8_wow_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_wow_bart-large_refine > eval_logs/threshold_0.9_wow_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_wow_bart-large_refine > eval_logs/threshold_1.0_wow_bart-large_refine.out &

##woner
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_woner_bart-large_refine > eval_logs/threshold_0.5_wow_woner_bart-large_refine.out &

CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_woner_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_wow_woner_bart-large_refine_only.out &


## BART-base
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_wow_bart-base_refine > eval_logs/threshold_0.0_wow_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_wow_bart-base_refine > eval_logs/threshold_0.1_wow_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_wow_bart-base_refine > eval_logs/threshold_0.2_wow_bart-base_refine.out &


#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_wow_bart-base_refine > eval_logs/threshold_0.3_wow_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_wow_bart-base_refine > eval_logs/threshold_0.4_wow_bart-base_refine.out &


#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_bart-base_refine > eval_logs/threshold_0.5_wow_bart-base_refine.out &


CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_wow_bart-base_refine_only.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_wow_bart-base_refine > eval_logs/threshold_0.6_wow_bart-base_refine.out &
#

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_wow_bart-base_refine > eval_logs/threshold_0.7_wow_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_wow_bart-base_refine > eval_logs/threshold_0.8_wow_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_wow_bart-base_refine > eval_logs/threshold_0.9_wow_bart-base_refine.out &


#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_wow_bart-base_refine > eval_logs/threshold_1.0_wow_bart-base_refine.out &

##woner
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/epoch6-02.4663.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_woner_bart-base_refine > eval_logs/threshold_0.5_wow_woner_bart-base_refine.out &

#woner
CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/epoch6-02.4663.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_woner_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_wow_woner_bart-base_refine_only.out &


#### EDMem prediction

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_wow_EDMem_refine > eval_logs/threshold_0.0_wow_EDMem_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_wow_EDMem_refine > eval_logs/threshold_0.3_wow_EDMem_refine.out &

#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_EDMem_refine > eval_logs/threshold_0.5_wow_EDMem_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_wow_EDMem_refine > eval_logs/threshold_0.7_wow_EDMem_refine.out &

#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_wow_EDMem_refine > eval_logs/threshold_1.0_wow_EDMem_refine.out &


##woner
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_sota/wow_dev_predictions_with_knowledge_NER_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_woner_EDMem_refine > eval_logs/threshold_0.5_wow_woner_EDMem_refine.out &
