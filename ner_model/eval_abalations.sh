########################################################################################################################
##############################################BART-BASE#################################################################


###BART base ner wow
## before refinebr
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_base_wow_ner_before_refine > eval_logs/bart_base_wow_ner_before_refine.out &

###b10
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b10 > eval_logs/bart_base_wow_ner_refine_b10.out &

###b1
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b1 > eval_logs/bart_base_wow_ner_refine_b1.out &

###b5
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b5 > eval_logs/bart_base_wow_ner_refine_b5.out &

####BART base ner cmudog
###before refine
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_base_cmu_ner_before_refine > eval_logs/bart_base_cmu_ner_before_refine.out &
#
###b10
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b10 > eval_logs/bart_base_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b1 > eval_logs/bart_base_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b5 > eval_logs/bart_base_cmu_ner_refine_b5.out &

####BART base ner focus
###before refine
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_base_focus_ner_before_refine > eval_logs/bart_base_focus_ner_before_refine.out &
#
###b10
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b10 > eval_logs/bart_base_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b1 > eval_logs/bart_base_focus_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b5 > eval_logs/bart_base_focus_ner_refine_b5.out &
#

########################################################################################################################
##############################################BART-LARGE#################################################################

####BART large ner wow
##b10
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b10 > eval_logs/bart_large_wow_ner_refine_b10.out &

###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b1 > eval_logs/bart_large_wow_ner_refine_b1.out &

###b5
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b5 > eval_logs/bart_large_wow_ner_refine_b5.out &

####BART large ner cmudog
###b10
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b10 > eval_logs/bart_large_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b1 > eval_logs/bart_large_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b5 > eval_logs/bart_large_cmu_ner_refine_b5.out &

#####BART large ner focus
###b10
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b10 > eval_logs/bart_large_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b1 > eval_logs/bart_large_focus_ner_refine_b1.out &

###b5
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b5 > eval_logs/bart_large_focus_ner_refine_b5.out &
#

########################################################################################################################
################################################T5-BASE#################################################################

####T5 base ner wow
###b10
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/epoch27-02.2912.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_base_wow_ner_refine_b10 > eval_logs/t5_base_wow_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type wow --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/epoch27-02.2912.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_base_wow_ner_refine_b1 > eval_logs/t5_base_wow_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type wow --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/epoch27-02.2912.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_base_wow_ner_refine_b5 > eval_logs/t5_base_wow_ner_refine_b5.out &
#
####T5 base ner cmudog
###b10
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/epoch53-02.9197.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_base_cmu_ner_refine_b10 > eval_logs/t5_base_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/epoch53-02.9197.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_base_cmu_ner_refine_b1 > eval_logs/t5_base_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/epoch53-02.9197.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_base_cmu_ner_refine_b5 > eval_logs/t5_base_cmu_ner_refine_b5.out &

####T5 base ner focus
###b10
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/epoch30-01.1982.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_base_focus_ner_refine_b10 > eval_logs/t5_base_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/epoch30-01.1982.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_base_focus_ner_refine_b1 > eval_logs/t5_base_focus_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model t5-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/epoch30-01.1982.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_base_focus_ner_refine_b5 > eval_logs/t5_base_focus_ner_refine_b5.out &


########################################################################################################################
################################################T5-LARGE################################################################


####T5 large ner wow
###b10
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_large_wow_ner_refine_b10 > eval_logs/t5_large_wow_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_large_wow_ner_refine_b1 > eval_logs/t5_large_wow_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type wow --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_large_wow_ner_refine_b5 > eval_logs/t5_large_wow_ner_refine_b5.out &

####T5 large ner cmudog
###b10
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/epoch7-03.3682.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_large_cmu_ner_refine_b10 > eval_logs/t5_large_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/epoch7-03.3682.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_large_cmu_ner_refine_b1 > eval_logs/t5_large_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/epoch7-03.3682.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_large_cmu_ner_refine_b5 > eval_logs/t5_large_cmu_ner_refine_b5.out &

####T5 large ner focus
###b10
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/epoch8-01.2280.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag t5_large_focus_ner_refine_b10 > eval_logs/t5_large_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/epoch8-01.2280.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag t5_large_focus_ner_refine_b1 > eval_logs/t5_large_focus_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model t5-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/epoch8-01.2280.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/t5_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag t5_large_focus_ner_refine_b5 > eval_logs/t5_large_focus_ner_refine_b5.out &


########################################################################################################################
#################################################WONER################################################################
########################################################################################################################


########################################################################################################################
################################################BART-BASE###############################################################
####BART base ner wow
###b10
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/new_test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/epoch6-02.4663.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b10 > eval_logs/woner_bart_base_wow_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/new_test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/epoch6-02.4663.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b1 > eval_logs/woner_bart_base_wow_ner_refine_b1.out &
#
##b5
CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/new_wo_NER/wow_base/gen1_ner0_E100_nops/epoch8-02.4147.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_ner_refine_b5 > eval_logs/woner_bart_base_wow_ner_refine_b5.out &

####BART base ner cmudog
###b10
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/epoch14-03.0137.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b10 > eval_logs/woner_bart_base_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/epoch14-03.0137.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b1 > eval_logs/woner_bart_base_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/epoch14-03.0137.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_ner_refine_b5 > eval_logs/woner_bart_base_cmu_ner_refine_b5.out &
#
####BART base ner focus
###b10
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wo_prompt_100/epoch11-01.2375.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b10 > eval_logs/woner_bart_base_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wo_prompt_100/epoch11-01.2375.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b1 > eval_logs/woner_bart_base_focus_ner_refine_b1.out &
#
###b5
CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/woner/wo_prompt_100/epoch9-01.2141.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wo_prompt_100/ \
--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_ner_refine_b5 > eval_logs/woner_bart_base_focus_ner_refine_b5.out &


########################################################################################################################
##############################################BART-LARGE#################################################################

####BART large ner wow
###b10
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/new_test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b10 > eval_logs/woner_bart_large_wow_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/new_test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b1 > eval_logs/woner_bart_large_wow_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type wow --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/new_test_random_split.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v3/our_test_random_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/epoch4-02.2932.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_ner_refine_b5 > eval_logs/woner_bart_large_wow_ner_refine_b5.out &
#
####BART large ner cmudog
###b10
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b10 > eval_logs/woner_bart_large_cmu_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b1 > eval_logs/woner_bart_large_cmu_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_ner_refine_b5 > eval_logs/woner_bart_large_cmu_ner_refine_b5.out &

####BART large ner focus
###b10
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch6-01.1220.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b10 > eval_logs/woner_bart_large_focus_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch6-01.1220.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b1 > eval_logs/woner_bart_large_focus_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch6-01.1220.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_ner_refine_b5 > eval_logs/woner_bart_large_focus_ner_refine_b5.out &
