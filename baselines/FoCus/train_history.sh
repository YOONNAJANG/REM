#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

#baselines

#LED aug data
#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
#--model_name LED \
#--model_path allenai/led-base-16384 \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 1 \
#--valid_batch_size 1 \
#--grad_accum 256 \
#--lr 5e-5 \
#--optimizer AdamW \
#--get_aug_data True \
#--flag Aug_woKL_DPRBi_LED \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/Aug_woKL_DPRBi_LED.log &

#T5 aug data
#CUDA_VISIBLE_DEVICES=3 nohup python train_focus.py \
#--model_name T5 \
#--model_path t5-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 4 \
#--valid_batch_size 2 \
#--grad_accum 64 \
#--lr 1e-4 \
#--optimizer Adafactor \
#--get_aug_data True \
#--flag Aug_woKL_DPRBi_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/Aug_woKL_DPRBi_T5.log &


#BART #augmented data with landmark info in input
#CUDA_VISIBLE_DEVICES=5 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--get_aug_data True \
#--flag Aug_woKL_DPRBi_BART_CLS_landmark \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/Aug_woKL_DPRBi_BART_CLS_landmark.log &




#BART KL2 original data with landmark info in input
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 2 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag KL2_DPRBi_BART_CLS_landmark \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/KL2_DPRBi_BART_CLS_landmark.log &



#BART KL2 augmented data with landmark info in input
#CUDA_VISIBLE_DEVICES=5 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 2 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--get_aug_data True \
#--flag Aug_KL2_DPRBi_BART_CLS_landmark \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/Aug_KL2_DPRBi_BART_CLS_landmark.log &













#CUDA_VISIBLE_DEVICES=2 nohup python train_focus.py \
#--model_name LED \
#--model_path allenai/led-base-16384 \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 1 \
#--valid_batch_size 1 \
#--grad_accum 256 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag KL_DPRBi_LED \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/KL_DPRBi_LED.log &


##knowledge selection CLS
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_DPRBi_CLS \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_DPRBi_CLS_BART.log &

#knowledge selection CLS landmarkname
#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_DPRBi_CLS_landmarkname \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_DPRBi_CLS_landmarkname_BART.log &



#add landmarkname to candidates
#CUDA_VISIBLE_DEVICES=3 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_DPRBi_CLS_landmarkname_cands \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_DPRBi_CLS_landmarkname_cands_BART.log &



#Train bart - maximum batch per 1 GPU: 8
#kl_coef는 0으로 설정, train_batch * grad_accum = 256
#
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type BM25 \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_BM25 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_BM25_BART.log &


#CUDA_VISIBLE_DEVICES=3 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type TFIDF \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_TFIDF \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_TFIDF_BART.log &


#baseline FoCus
#CUDA_VISIBLE_DEVICES=2 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type TFIDF \
#--ps_coef 1 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 10 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 6.25e-5 \
#--optimizer AdamW \
#--flag FoCus_baseline_BART \
#--n_epochs 2 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/FoCus_baseline_BART.log &


#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type TFIDF_sen \
#--ps_coef 2 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag knowledge_selection_TFIDFsen \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/knowledge_selection_TFIDFsen_BART.log &

#CUDA_VISIBLE_DEVICES=3 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--ps_coef 1 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag find_lambda_1_1_5 \
#--n_epochs 5 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/find_lambda_1_1_5_BART.log &
#
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/focus_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type DPR \
#--ps_coef 1 \
#--kl_coef 0 \
#--kn_coef 1 \
#--lm_coef 10 \
#--train_batch_size 8 \
#--valid_batch_size 2 \
#--grad_accum 32 \
#--lr 5e-5 \
#--optimizer AdamW \
#--flag find_lambda_1_1_10 \
#--n_epochs 5 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log/find_lambda_1_1_10_BART.log &




echo





