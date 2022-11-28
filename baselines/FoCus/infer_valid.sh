#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.



##test knowledge_selection_DPRBi_CLS_landmarkname_cands_BART
CUDA_VISIBLE_DEVICES=2 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_DPRBi_CLS_landmarkname_cands/epoch6-ppl3.8580.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART_greedy.log &&



##test KL2_DPRBi_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=2 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.8777.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL2_DPRBi_BART_CLS_landmark_cands_greedy.log &&


##Aug_woKL_DPRBi_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=2 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.3516.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_woKL_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_woKL_DPRBi_BART_CLS_landmark_cands_greedy.log &&



##Aug_KL2_DPRBi_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=2 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_BART_CLS_landmark_cands/epoch9-ppl3.9081.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL2_DPRBi_BART_CLS_landmark_cands_greedy.log &&


##KL0_DPRBi_T5_greedy
CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_DPRBi_T5/epoch4-ppl3.8025.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL0_DPRBi_T5_greedy.log &&



##KL2_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_DPRBi_T5/epoch5-ppl3.5424.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL2_DPRBi_T5_greedy.log &&




##Aug_KL0_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_DPRBi_T5/epoch4-ppl4.1543.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL0_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL0_DPRBi_T5_greedy.log &&



##Aug_KL2_DPRBi_T5 greedy
CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type DPR \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_DPRBi_T5/epoch3-ppl3.5069.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_DPRBi_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL2_DPRBi_T5_greedy.log &&


#knowledge_selection_BM25_CLS_landmarkname_cands_BART greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/knowledge_selection_BMpar_CLS_landmarkname_cands_BART/epoch6-ppl3.8456.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag knowledge_selection_BM25_CLS_landmarkname_cands_BART_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/knowledge_selection_BM25_CLS_landmarkname_cands_BART_greedy.log &&


#KL2_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BMpar_BART_CLS_landmark_cands/epoch9-ppl3.3754.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL2_BM25_BART_CLS_landmark_cands_greedy.log &&



#Aug_woKL_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_woKL_BM25_BART_CLS_landmark_cands/epoch9-ppl3.4276.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_woKL_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_woKL_BM25_BART_CLS_landmark_cands_greedy.log &&


#Aug_KL2_BM25_BART_CLS_landmark_cands greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_BART_CLS_landmark_cands/epoch9-ppl3.3563.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_BM25_BART_CLS_landmark_cands_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL2_BM25_BART_CLS_landmark_cands_greedy.log &&


#KL0_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL0_BM25_T5/epoch5-ppl3.5815.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL0_BM25_T5_greedy.log &&


#KL2_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/KL2_BM25_T5/epoch5-ppl4.1936.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL2_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/KL2_BM25_T5_greedy.log &&


#Aug_KL0_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL0_BM25_T5/epoch4-ppl3.5988.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL0_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL0_BM25_T5_greedy.log &&


#Aug_KL2_BM25_T5 greedy
CUDA_VISIBLE_DEVICES=0 nohup python infer_valid.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache_valid_only.tar.gz \
--model_name T5 \
--model_path t5-base \
--retrieval_type BM25 \
--checkpoint /home/mnt/yoonna/focus_modeling/model/Aug_KL2_BM25_T5/epoch3-ppl3.8816.ckpt \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag Aug_KL2_BM25_T5_greedy \
--output_dir /home/mnt/yoonna/focus_modeling/valid_output_new/ > valid_log/Aug_KL2_BM25_T5_greedy.log &&


echo
