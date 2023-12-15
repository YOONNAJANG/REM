##KL0_DPRBi_BART_greedy
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl7.9310.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL0_DPRBi_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_BART_greedy.log &&
#
##KL0_DPRBi_BART_top5
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl7.9310.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL0_DPRBi_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_BART_top5.log &&
#
##KL0_DPRBi_BART_top10
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl7.9310.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL0_DPRBi_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_BART_top10.log &&
#
#
##KL0_DPRBi_BART_beam2
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl7.9310.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL0_DPRBi_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_BART_beam2.log &&

##KL0_DPRBi_BART_beam5
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl7.9310.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL0_DPRBi_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_BART_beam5.log &&

#KL0_BM25_BART_greedy
CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_BART/epoch9-ppl7.9051.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_BM25_BART_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_BART_greedy.log &&

##KL0_BM25_BART_top5
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_BART/epoch9-ppl7.9051.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL0_BM25_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_BART_top5.log &&
#
##KL0_BM25_BART_top10
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_BART/epoch9-ppl7.9051.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL0_BM25_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_BART_top10.log &&
#
##KL0_BM25_BART_beam2
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_BART/epoch9-ppl7.9051.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL0_BM25_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_BART_beam2.log &&
#
##KL0_BM25_BART_beam5
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_BART/epoch9-ppl7.9051.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL0_BM25_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_BART_beam5.log &&
#
#
##KL2_DPRBi_BART_greedy
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART/epoch9-ppl7.6251.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL2_DPRBi_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_greedy.log &&
#
##KL2_DPRBi_BART_top5
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART/epoch9-ppl7.6251.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL2_DPRBi_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_top5.log &&
#
##KL2_DPRBi_BART_top10
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART/epoch9-ppl7.6251.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL2_DPRBi_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_top10.log &&
#
#
##KL2_DPRBi_BART_beam2
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART/epoch9-ppl7.6251.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL2_DPRBi_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_beam2.log &&
#
##KL2_DPRBi_BART_beam5
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART/epoch9-ppl7.6251.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL2_DPRBi_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_beam5.log &&
#
##KL2_BM25_BART_greedy
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_BART/epoch9-ppl7.6383.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL2_BM25_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_greedy.log &&
#
##KL2_BM25_BART_top5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_BART/epoch9-ppl7.6383.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL2_BM25_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_top5.log &&
#
##KL2_BM25_BART_top10
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_BART/epoch9-ppl7.6383.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL2_BM25_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_top10.log &&
#
##KL2_BM25_BART_beam2
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_BART/epoch9-ppl7.6383.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL2_BM25_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_beam2.log &&
#
##KL2_BM25_BART_beam5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_BART/epoch9-ppl7.6383.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL2_BM25_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_beam5.log &&
#
#
##Aug_KL0_DPRBi_BART_greedy
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_BART/epoch9-ppl8.0417.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL0_DPRBi_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_BART_greedy.log &&
#
##Aug_KL0_DPRBi_BART_top5
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_BART/epoch9-ppl8.0417.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL0_DPRBi_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_BART_top5.log &&
#
##Aug_KL0_DPRBi_BART_top10
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_BART/epoch9-ppl8.0417.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL0_DPRBi_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_BART_top10.log &&
#
#
##Aug_KL0_DPRBi_BART_beam2
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_BART/epoch9-ppl8.0417.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL0_DPRBi_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_BART_beam2.log &&
#
##Aug_KL0_DPRBi_BART_beam5
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_BART/epoch9-ppl8.0417.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL0_DPRBi_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_BART_beam5.log &&
#
##Aug_KL0_BM25_BART_greedy
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_BART/epoch9-ppl8.1472.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL0_BM25_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_BART_greedy.log &&
#
##Aug_KL0_BM25_BART_top5
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_BART/epoch9-ppl8.1472.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL0_BM25_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_BART_top5.log &&
#
##Aug_KL0_BM25_BART_top10
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_BART/epoch9-ppl8.1472.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL0_BM25_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_BART_top10.log &&
#
##Aug_KL0_BM25_BART_beam2
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_BART/epoch9-ppl8.1472.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL0_BM25_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_BART_beam2.log &&
#
##Aug_KL0_BM25_BART_beam5
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_BART/epoch9-ppl8.1472.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL0_BM25_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_BART_beam5.log &&
#
#
##Aug_KL2_DPRBi_BART_greedy
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART/epoch9-ppl7.7324.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_greedy.log &&
#
##Aug_KL2_DPRBi_BART_top5
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART/epoch9-ppl7.7324.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL2_DPRBi_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_top5.log &&
#
##Aug_KL2_DPRBi_BART_top10
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART/epoch9-ppl7.7324.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL2_DPRBi_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_top10.log &&
#
#
##Aug_KL2_DPRBi_BART_beam2
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART/epoch9-ppl7.7324.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL2_DPRBi_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_beam2.log &&
#
##Aug_KL2_DPRBi_BART_beam5
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART/epoch9-ppl7.7324.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL2_DPRBi_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_beam5.log &&
#
##Aug_KL2_BM25_BART_greedy
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART/epoch9-ppl7.8077.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_BM25_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_greedy.log &&
#
##Aug_KL2_BM25_BART_top5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART/epoch9-ppl7.8077.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL2_BM25_BART_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_top5.log &&
#
##Aug_KL2_BM25_BART_top10
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART/epoch9-ppl7.8077.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL2_BM25_BART_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_top10.log &&
#
##Aug_KL2_BM25_BART_beam2
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART/epoch9-ppl7.8077.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL2_BM25_BART_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_beam2.log &&
#
##Aug_KL2_BM25_BART_beam5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART/epoch9-ppl7.8077.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL2_BM25_BART_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_beam5.log &&
#
#
#
#
###########################################################T5###########################################################3
#
#
#
#
#
##KL0_DPRBi_T5_greedy
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch5-ppl9.5873.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL0_DPRBi_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_greedy.log &&
#
##KL0_DPRBi_T5_top5
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch5-ppl9.5873.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL0_DPRBi_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_top5.log &&
#
##KL0_DPRBi_T5_top10
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch5-ppl9.5873.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL0_DPRBi_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_top10.log &&
#
#
##KL0_DPRBi_T5_beam2
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch5-ppl9.5873.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL0_DPRBi_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_beam2.log &&
#
##KL0_DPRBi_T5_beam5
#CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch5-ppl9.5873.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL0_DPRBi_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_beam5.log &&
#
#
##KL0_BM25_T5_greedy
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch6-ppl9.6221.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL0_BM25_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_greedy.log &&
#
##KL0_BM25_T5_top5
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch6-ppl9.6221.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL0_BM25_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_top5.log &&
#
##KL0_BM25_T5_top10
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch6-ppl9.6221.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL0_BM25_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_top10.log &&
#
##KL0_BM25_T5_beam2
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch6-ppl9.6221.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL0_BM25_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_beam2.log &&
#
##KL0_BM25_T5_beam5
#CUDA_VISIBLE_DEVICES=5 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch6-ppl9.6221.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL0_BM25_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_beam5.log &&
#
#
##KL2_DPRBi_T5_greedy
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl9.7607.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL2_DPRBi_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_greedy.log &&
#
##KL2_DPRBi_T5_top5
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl9.7607.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL2_DPRBi_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_top5.log &&
#
##KL2_DPRBi_T5_top10
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl9.7607.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL2_DPRBi_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_top10.log &&
#
#
##KL2_DPRBi_T5_beam2
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl9.7607.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL2_DPRBi_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_beam2.log &&
#
##KL2_DPRBi_T5_beam5
#CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl9.7607.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL2_DPRBi_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_beam5.log &&
#
#
##KL2_BM25_T5_greedy
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl9.9517.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL2_BM25_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_greedy.log &&
#
##KL2_BM25_T5_top5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl9.9517.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag KL2_BM25_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_top5.log &&
#
##KL2_BM25_T5_top10
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl9.9517.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag KL2_BM25_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_top10.log &&
#
##KL2_BM25_T5_beam2
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl9.9517.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag KL2_BM25_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_beam2.log &&
#
##KL2_BM25_T5_beam5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl9.9517.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag KL2_BM25_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_beam5.log &&
#
##Aug_KL0_DPRBi_T5_greedy
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch5-ppl10.3286.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL0_DPRBi_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_greedy.log &&
#
##Aug_KL0_DPRBi_T5_top5
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch5-ppl10.3286.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL0_DPRBi_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_top5.log &&
#
##Aug_KL0_DPRBi_T5_top10
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch5-ppl10.3286.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL0_DPRBi_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_top10.log &&
#
#
##Aug_KL0_DPRBi_T5_beam2
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch5-ppl10.3286.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL0_DPRBi_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_beam2.log &&
#
##Aug_KL0_DPRBi_T5_beam5
#CUDA_VISIBLE_DEVICES=0 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch5-ppl10.3286.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL0_DPRBi_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_beam5.log &&
#
##Aug_KL0_BM25_T5_greedy
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch5-ppl9.8590.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL0_BM25_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_greedy.log &&
#
##Aug_KL0_BM25_T5_top5
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch5-ppl9.8590.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL0_BM25_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_top5.log &&
#
##Aug_KL0_BM25_T5_top10
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch5-ppl9.8590.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL0_BM25_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_top10.log &&
#
##Aug_KL0_BM25_T5_beam2
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch5-ppl9.8590.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL0_BM25_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_beam2.log &&
#
##Aug_KL0_BM25_T5_beam5
#CUDA_VISIBLE_DEVICES=1 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch5-ppl9.8590.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL0_BM25_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_beam5.log &&
#
#
#
##Aug_KL2_DPRBi_T5_greedy
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch5-ppl9.9984.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_greedy.log &&
#
##Aug_KL2_DPRBi_T5_top5
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch5-ppl9.9984.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL2_DPRBi_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_top5.log &&
#
##Aug_KL2_DPRBi_T5_top10
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch5-ppl9.9984.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL2_DPRBi_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_top10.log &&
#
#
##Aug_KL2_DPRBi_T5_beam2
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch5-ppl9.9984.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL2_DPRBi_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_beam2.log &&
#
##Aug_KL2_DPRBi_T5_beam5
#CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch5-ppl9.9984.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL2_DPRBi_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_beam5.log &&
#
#
##Aug_KL2_BM25_T5_greedy
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch5-ppl10.0714.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_BM25_T5_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_greedy.log &&
#
##Aug_KL2_BM25_T5_top5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch5-ppl10.0714.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 5 \
#--flag Aug_KL2_BM25_T5_top5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_top5.log &&
#
##Aug_KL2_BM25_T5_top10
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch5-ppl10.0714.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--top_k 10 \
#--flag Aug_KL2_BM25_T5_top10 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_top10.log &&
#
##Aug_KL2_BM25_T5_beam2
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch5-ppl10.0714.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 2 \
#--flag Aug_KL2_BM25_T5_beam2 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_beam2.log &&
#
##Aug_KL2_BM25_T5_beam5
#CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch5-ppl10.0714.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 5 \
#--flag Aug_KL2_BM25_T5_beam5 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_beam5.log &&

#echo
