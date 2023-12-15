#FoCus
#CUDA_VISIBLE_DEVICES=4  python src/eval_refiner.py --data_type focus --pretrained_model facebook/bart-large \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner.json \
#--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/FoCus/output/2023_emnlp/bart_base/test_beam5_09k_B_B_with_ner_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_bart-large_refine > eval_logs/threshold_0.5_focus_bart-large_refine.out &


#CMUDoG
CUDA_VISIBLE_DEVICES=4 nohup python src/eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_bart-large_refine > eval_logs/threshold_0.5_cmu_bart-large_refine.out &


#WoW
CUDA_VISIBLE_DEVICES=3 nohup python src/eval_refiner.py --data_type wow --pretrained_model facebook/bart-large \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner.json \
--test_dataset_cache /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_with_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_wow_bart-large_refine > eval_logs/threshold_0.5_wow_bart-large_refine.out &