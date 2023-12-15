#KL0_BM25_BART_greedy
CUDA_VISIBLE_DEVICES=5 nohup python evaluate_test_yoonna.py \
--test_dataset_path /home/data/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/data/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/data/yoonna/Refiner/baselines/FoCus/ckpt/epoch9-ppl7.9051.ckpt \
--landmark_dic /home/data/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_BM25_BART_greedy \
--output_dir /home/data/yoonna/Refiner/baselines/FoCus/ckpt/ > test_log/KL0_BM25_BART_greedy.log &
