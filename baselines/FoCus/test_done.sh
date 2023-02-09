#knowledge_selection_BM25_CLS_landmarkname_cands_BART greedy
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_greedy.log &&

#knowledge_selection_BM25_CLS_landmarkname_cands_BART top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_top5.log &&

#knowledge_selection_BM25_CLS_landmarkname_cands_BART top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_top10.log &&

#knowledge_selection_BM25_CLS_landmarkname_cands_BART beam2
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_beam2.log &&

#knowledge_selection_BM25_CLS_landmarkname_cands_BART beam5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_beam5.log &&



#test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_greedy.log &&


##test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_top5.log &&

#running
#test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_top10.log &&


#running
#test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART beam2
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_beam2.log &&

#test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART beam5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_beam5.log &&




#KL2_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_CLS_landmark_cands_greedy.log &&


#KL2_BM25_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL2_BM25_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_CLS_landmark_cands_top5.log &&


#KL2_BM25_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL2_BM25_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_CLS_landmark_cands_top10.log &&


#KL2_BM25_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL2_BM25_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_CLS_landmark_cands_beam2.log &&


#KL2_BM25_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL2_BM25_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_BART_CLS_landmark_cands_beam5.log &&




#test KL2_DPRBi_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_CLS_landmark_cands_greedy.log &&

#running
#test KL2_DPRBi_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_CLS_landmark_cands_top5.log &&

#ready
#test KL2_DPRBi_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_CLS_landmark_cands_top10.log &&

#test KL2_DPRBi_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_CLS_landmark_cands_beam2.log &&


#test KL2_DPRBi_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_BART_CLS_landmark_cands_beam5.log &&



#Aug_woKL_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_greedy.log &&

#Aug_woKL_BM25_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_top5.log &&

#Aug_woKL_BM25_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_top10.log &&


#Aug_woKL_BM25_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_beam2.log &&

#Aug_woKL_BM25_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_beam5.log &&



#Aug_woKL_DPRBi_BART_CLS_landmark_cands
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_greedy.log &&




#Aug_woKL_DPRBi_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_top5.log &&


#Aug_woKL_DPRBi_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_top10.log &&


#Aug_woKL_DPRBi_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_beam2.log &&


#Aug_woKL_DPRBi_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_beam5.log &&




#Aug_KL2_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_greedy.log &&


#Aug_KL2_BM25_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_top5.log &&


#Aug_KL2_BM25_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_top10.log &&

#Aug_KL2_BM25_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_beam2.log &&

#Aug_KL2_BM25_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_beam5.log &&





#Aug_KL2_DPRBi_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy.log &&


#Aug_KL2_DPRBi_BART_CLS_landmark_cands top5
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_top5.log &&


#Aug_KL2_DPRBi_BART_CLS_landmark_cands top10
CUDA_VISIBLE_DEVICES=6 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_top10.log &&

#Aug_KL2_DPRBi_BART_CLS_landmark_cands beam2
CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_beam2.log &&


#Aug_KL2_DPRBi_BART_CLS_landmark_cands beam5
CUDA_VISIBLE_DEVICES=2 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_beam5.log &&



#KL0_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_greedy.log &&

#KL0_BM25_T5 top5
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL0_BM25_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_top5.log &&

#KL0_BM25_T5 top10
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL0_BM25_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_top10.log &&

#KL0_BM25_T5 beam2
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL0_BM25_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_beam2.log &&

#KL0_BM25_T5 beam5
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL0_BM25_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_BM25_T5_beam5.log &&


#KL0_DPRBi_T5_greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_greedy.log &&

#KL0_DPRBi_T5_top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL0_DPRBi_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_top5.log &&


#KL0_DPRBi_T5_top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL0_DPRBi_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_top10.log &&


#KL0_DPRBi_T5_beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL0_DPRBi_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_beam2.log &&


#KL0_DPRBi_T5_beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL0_DPRBi_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL0_DPRBi_T5_beam5.log &&


#KL2_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_greedy.log &&


#KL2_BM25_T5 top5
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL2_BM25_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_top5.log &&

#KL2_BM25_T5 top10
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL2_BM25_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_top10.log &&

#KL2_BM25_T5 beam2
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL2_BM25_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_beam2.log &&

#KL2_BM25_T5 beam5
CUDA_VISIBLE_DEVICES=4 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL2_BM25_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_BM25_T5_beam5.log &&





#KL2_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_greedy.log &&

#KL2_DPRBi_T5 top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag KL2_DPRBi_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_top5.log &&


#KL2_DPRBi_T5 top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag KL2_DPRBi_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_top10.log &&


#KL2_DPRBi_T5 beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag KL2_DPRBi_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_beam2.log &&



#KL2_DPRBi_T5 beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag KL2_DPRBi_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/KL2_DPRBi_T5_beam5.log &&



#Aug_KL0_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL0_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_greedy.log &&

#Aug_KL0_BM25_T5 top5
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL0_BM25_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_top5.log &&


#Aug_KL0_BM25_T5 top10
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL0_BM25_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_top10.log &&

#Aug_KL0_BM25_T5 beam2
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL0_BM25_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_beam2.log &&

#Aug_KL0_BM25_T5 beam5
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL0_BM25_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_BM25_T5_beam5.log &&




#Aug_KL0_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL0_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_greedy.log &&


#Aug_KL0_DPRBi_T5 top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL0_DPRBi_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_top5.log &&


#Aug_KL0_DPRBi_T5 top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL0_DPRBi_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_top10.log &&


#Aug_KL0_DPRBi_T5 beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL0_DPRBi_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_beam2.log &&


#Aug_KL0_DPRBi_T5 beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL0_DPRBi_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL0_DPRBi_T5_beam5.log &&


#Aug_KL2_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_greedy.log &&

#Aug_KL2_BM25_T5 top5
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL2_BM25_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_top5.log &&

#Aug_KL2_BM25_T5 top10
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL2_BM25_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_top10.log &&

#Aug_KL2_BM25_T5 beam2
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL2_BM25_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_beam2.log &&

#Aug_KL2_BM25_T5 beam5
CUDA_VISIBLE_DEVICES=3 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL2_BM25_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_beam5.log &&



#Aug_KL2_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_greedy.log &&

#Aug_KL2_DPRBi_T5 top5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 5 \
--flag Aug_KL2_DPRBi_T5_top5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_top5.log &&


#Aug_KL2_DPRBi_T5 top10
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--top_k 10 \
--flag Aug_KL2_DPRBi_T5_top10 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_top10.log &&


#Aug_KL2_DPRBi_T5 beam2
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 2 \
--flag Aug_KL2_DPRBi_T5_beam2 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_beam2.log &&

#Aug_KL2_DPRBi_T5 beam5
CUDA_VISIBLE_DEVICES=7 python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 5 \
--flag Aug_KL2_DPRBi_T5_beam5 \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_beam5.log &&






##FoCus_baseline_BART
#CUDA_VISIBLE_DEVICES=4 nohup python evaluate_test_prev.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type TFIDF \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/FoCus_baseline_BART/epoch1-ppl6.1085.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag FoCus_baseline_BART_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/FoCus_baseline_BART_greedy.log &&


#Aug_KL2_DPRBi_BART_CLS_landmark_pos
#CUDA_VISIBLE_DEVICES=2 nohup python evaluate_test_prev.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_pos0/epoch9-ppl3.3788.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy_pos0 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy_pos0.log &



##Aug_KL2_DPRBi_T5_pos0
#CUDA_VISIBLE_DEVICES=5 nohup python evaluate_test_prev.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5_pos0/epoch4-ppl3.4815.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_T5_greedy_pos0 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_greedy_pos0.log &

##Aug_KL2_DPRBi_BART_CLS_landmark_cands greedy qr
#CUDA_VISIBLE_DEVICES=4 nohup python evaluate_test_prev.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--question_rewrite True \
#--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy_qr \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/qr_Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy.log &&


##Aug_KL2_DPRBi_T5 greedy qr
#CUDA_VISIBLE_DEVICES=4 nohup python evaluate_test_prev.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--question_rewrite True \
#--flag Aug_KL2_DPRBi_T5_greedy_qr \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/qr_Aug_KL2_DPRBi_T5_greedy.log &&

#
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_BM25_BART_CLS_landmark_cands_greedy \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_BART_CLS_landmark_cands_greedy.log &&

##Aug_woKL_BM25_BART_CLS_landmark_cands greedy fortest
#CUDA_VISIBLE_DEVICES=1 nohup python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_woKL_BM25_BART_CLS_landmark_cands_greedy_fortest \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_woKL_BM25_BART_CLS_landmark_cands_greedy_fortest.log &&


##Aug_KL2_DPRBi_BART_CLS_landmark_cands greedy pos0
#CUDA_VISIBLE_DEVICES=2 nohup python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_pos0/epoch9-ppl3.8822.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy_pos0 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy_pos0.log &


##Aug_KL2_DPRBi_T5 greedy pos0
#CUDA_VISIBLE_DEVICES=1 nohup python evaluate_test.py \
#--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--model_name T5 \
#--model_path t5-base \
#--retrieval_type DPR \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5_pos0/epoch5-ppl3.9892.ckpt \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag Aug_KL2_DPRBi_T5_greedy_pos0 \
#--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_DPRBi_T5_greedy_pos0.log &


#Aug_KL2_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/test_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/output/ > test_log/Aug_KL2_BM25_T5_greedy.log &

echo