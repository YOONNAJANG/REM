###bart-base prediction
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_focus_bart-base_refine > eval_logs/threshold_0.0_focus_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_focus_bart-base_refine > eval_logs/threshold_0.1_focus_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_focus_bart-base_refine > eval_logs/threshold_0.2_focus_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_focus_bart-base_refine > eval_logs/threshold_0.3_focus_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_focus_bart-base_refine > eval_logs/threshold_0.4_focus_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_bart-base_refine > eval_logs/threshold_0.5_focus_bart-base_refine.out &



CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_focus_bart-base_refine_only.out &



#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_focus_bart-base_refine > eval_logs/threshold_0.6_focus_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_focus_bart-base_refine > eval_logs/threshold_0.7_focus_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_focus_bart-base_refine > eval_logs/threshold_0.8_focus_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_focus_bart-base_refine > eval_logs/threshold_0.9_focus_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_focus_bart-base_refine > eval_logs/threshold_1.0_focus_bart-base_refine.out &

##woner
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wo_prompt_100/epoch9-01.2141.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_woner_bart-base_refine > eval_logs/threshold_0.5_focus_woner_bart-base_refine.out &


##woner
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/wo_prompt_100/epoch9-01.2141.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_woner_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_focus_woner_bart-base_refine_only.out &
#


###bart-large prediction
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_focus_bart-large_refine > eval_logs/threshold_0.0_focus_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_focus_bart-large_refine > eval_logs/threshold_0.1_focus_bart-large_refine.out &
#
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_focus_bart-large_refine > eval_logs/threshold_0.2_focus_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_focus_bart-large_refine > eval_logs/threshold_0.3_focus_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_focus_bart-large_refine > eval_logs/threshold_0.4_focus_bart-large_refine.out &


#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_bart-large_refine > eval_logs/threshold_0.5_focus_bart-large_refine.out &

#
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_focus_bart-large_refine_only.out &




#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_focus_bart-large_refine > eval_logs/threshold_0.6_focus_bart-large_refine.out &


#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_focus_bart-large_refine > eval_logs/threshold_0.7_focus_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_focus_bart-large_refine > eval_logs/threshold_0.8_focus_bart-large_refine.out &
#
#
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_focus_bart-large_refine > eval_logs/threshold_0.9_focus_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_focus_bart-large_refine > eval_logs/threshold_1.0_focus_bart-large_refine.out &

##woner
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch8-01.0974.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_woner_bart-large_refine > eval_logs/threshold_0.5_focus_woner_bart-large_refine.out &


#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch8-01.0974.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_woner_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_focus_woner_bart-large_refine_only.out &
#
#


##INFO prediction
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_focus_INFO_refine > eval_logs/threshold_0.0_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_focus_INFO_refine > eval_logs/threshold_0.1_focus_INFO_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_focus_INFO_refine > eval_logs/threshold_0.2_focus_INFO_refine.out &

#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_focus_INFO_refine > eval_logs/threshold_0.3_focus_INFO_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_INFO_refine > eval_logs/threshold_0.5_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_focus_INFO_refine > eval_logs/threshold_0.7_focus_INFO_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_focus_INFO_refine > eval_logs/threshold_1.0_focus_INFO_refine.out &



#beam10
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_focus_INFO_refine_b10 > eval_logs/threshold_0.1_focus_INFO_refine_b10.out &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_focus_INFO_refine_b10 > eval_logs/threshold_0.2_focus_INFO_refine_b10.out &

#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_focus_INFO_refine_b10 > eval_logs/threshold_0.3_focus_INFO_refine_b10.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_INFO_refine_b10 > eval_logs/threshold_0.5_focus_INFO_refine_b10.out &

##woner
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/epoch8-01.0974.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_woner_INFO_refine_b10 > eval_logs/threshold_0.5_focus_woner_INFO_refine_b10.out &
