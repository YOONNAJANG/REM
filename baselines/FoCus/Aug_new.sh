##BART #augmented data with landmark info in input

#
##augmented data with landmark info in input & candidates
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py \
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
#--flag NEW_Aug_woKL_DPRBi_BART_CLS_landmark_cands \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_woKL_DPRBi_BART_CLS_landmark_cands.log &
#
#

#
#
#
###KL2 augmented data with landmark info in input & candidates
#CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py \
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
#--flag NEW_Aug_KL2_DPRBi_BART_CLS_landmark_cands \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_DPRBi_BART_CLS_landmark_cands.log &



##T5 aug data
#CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py \
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
#--flag NEW_Aug_woKL_DPRBi_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_woKL_DPRBi_T5.log &
#
##T5 KL Experiment - augmented kl0
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py \
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
#--flag NEW_Aug_KL0_DPRBi_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL0_DPRBi_T5.log &
#
##T5 KL Experiment - augmented kl2
#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
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
#--kl_coef 2 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 4 \
#--valid_batch_size 2 \
#--grad_accum 64 \
#--lr 1e-4 \
#--optimizer Adafactor \
#--get_aug_data True \
#--flag NEW_Aug_KL2_DPRBi_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_DPRBi_T5.log &


##Aug_woKL_BM25_BART_CLS_landmark_cands
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type BM25 \
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
#--flag NEW_Aug_woKL_BM25_BART_CLS_landmark_cands \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_woKL_BM25_BART_CLS_landmark_cands.log &

##Aug_KL2_BM25_BART_CLS_landmark_cands
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type BM25 \
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
#--flag NEW_Aug_KL2_BM25_BART_CLS_landmark_cands \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_BM25_BART_CLS_landmark_cands.log &






##Aug_KL0_BM25_T5
#CUDA_VISIBLE_DEVICES=4 nohup python train_focus.py \
#--model_name T5 \
#--model_path t5-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type BM25 \
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
#--flag NEW_Aug_KL0_BM25_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL0_BM25_T5.log &




##Aug_KL2_BM25_T5
#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
#--model_name T5 \
#--model_path t5-base \
#--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
#--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
#--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
#--landmark_dic ./retrieval/all_landmark_dic.json \
#--retrieval_type BM25 \
#--use_knowledge_embedidngs \
#--ps_coef 2 \
#--kl_coef 2 \
#--kn_coef 1 \
#--lm_coef 5 \
#--train_batch_size 4 \
#--valid_batch_size 2 \
#--grad_accum 64 \
#--lr 1e-4 \
#--optimizer Adafactor \
#--get_aug_data True \
#--flag NEW_Aug_KL2_BM25_T5 \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_BM25_T5.log &


#Aug_KL2_BM25_BART_CLS_landmark_cands pos0
CUDA_VISIBLE_DEVICES=0 nohup python train_focus.py \
--model_name BART \
--model_path facebook/bart-base \
--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--retrieval_type DPR \
--use_knowledge_embedidngs \
--ps_coef 2 \
--kl_coef 2 \
--kn_coef 1 \
--lm_coef 5 \
--train_batch_size 8 \
--valid_batch_size 2 \
--grad_accum 32 \
--lr 5e-5 \
--optimizer AdamW \
--get_aug_data True \
--flag NEW_Aug_KL2_DPRBi_BART_CLS_landmark_pos0 \
--n_epochs 10 \
--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_DPRBi_BART_CLS_landmark_pos0.log &
#DONE posweight 바꾸기

#Aug_KL2_DPRBi_T5 POS 0
CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
--model_name T5 \
--model_path t5-base \
--train_dataset_path /home/mnt/yoonna/focus_modeling/our_data/train_ours_augmented.json \
--train_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--dev_dataset_path /home/mnt/yoonna/focus_modeling/our_data/valid_ours_augmented.json \
--dev_dataset_cache /home/mnt/yoonna/focus_modeling/our_data/ours_cache.tar.gz \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--retrieval_type DPR \
--use_knowledge_embedidngs \
--ps_coef 2 \
--kl_coef 2 \
--kn_coef 1 \
--lm_coef 5 \
--train_batch_size 4 \
--valid_batch_size 2 \
--grad_accum 64 \
--lr 1e-4 \
--optimizer Adafactor \
--get_aug_data True \
--flag NEW_Aug_KL2_DPRBi_T5_pos0 \
--n_epochs 10 \
--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_DPRBi_T5_pos0.log &
#DONE posweight 바꾸기


#CUDA_VISIBLE_DEVICES=1 nohup python train_focus.py \
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
#--flag NEW_Aug_woKL_DPRBi_BART_CLS_landmark \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_woKL_DPRBi_BART_CLS_landmark.log &

##BART KL2 augmented data with landmark info in input
#CUDA_VISIBLE_DEVICES=7 nohup python train_focus.py \
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
#--flag NEW_Aug_KL2_DPRBi_BART_CLS_landmark \
#--n_epochs 10 \
#--output_dir /home/mnt/yoonna/focus_modeling/model/ > train_log_Aug/Aug_KL2_DPRBi_BART_CLS_landmark.log &

################################DONE!!!#############################################################################################################





echo