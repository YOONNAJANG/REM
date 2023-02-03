#CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/our_data/test_ours.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/our_data/infer_test_ours_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--retrieval_type BM25 \
#--checkpoint /home/data/ssh5131/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl8.6583.ckpt \
#--landmark_dic /home/data/ssh5131/focus_modeling/our_data/all_landmark_dic.json \
#--num_beams 1 \
#--flag KL0_BM25_BART_test \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/ > KL0_BM25_BART_test.out &
###> test_log/Aug_KL0_BM25_BART_top5.log &&

CUDA_VISIBLE_DEVICES=4 nohup python infer_valid.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/our_data/train_ours.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/our_data/infer_train_ours_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--retrieval_type BM25 \
--checkpoint /home/data/ssh5131/focus_modeling/model/KL0_DPRBi_BART/epoch9-ppl8.6583.ckpt \
--landmark_dic /home/data/ssh5131/focus_modeling/our_data/all_landmark_dic.json \
--num_beams 1 \
--flag KL0_BM25_BART_dev \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/ > KL0_BM25_BART_train.out &