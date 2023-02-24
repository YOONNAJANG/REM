
##########################################################
#################       FoCus      #######################
##########################################################
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 1 \
#--flag gen1_ner0.7_E100_greedy \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/ > test_log/gen1_ner0.7_E100_greedy.out &
###
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 1 \
#--top_k 5 \
#--flag gen1_ner0.7_E100_top5 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/  > test_log/gen1_ner0.7_E100_top5.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 1 \
#--top_k 10 \
#--flag gen1_ner0.7_E100_top10 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/ > test_log/gen1_ner0.7_E100_top10.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 2 \
#--flag gen1_ner0.7_E100_beam2 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/ > test_log/gen1_ner0.7_E100_beam2.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 5 \
#--flag gen1_ner0.7_E100_beam5 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/ > test_log/gen1_ner0.7_E100_beam5.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type focus \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus_new/gen1_ner0.7_E100/epoch25-valid_lm_loss1.4795.ckpt \
#--num_beams 10 \
#--flag gen1_ner0.7_E100_beam10 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/ > test_log/gen1_ner0.7_E100_beam10.out &


##########################################################
#################       WoW      #######################
###########################################################
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 1 \
#--flag gen1_ner0.3_E100_greedy \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_greedy.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 1 \
#--top_k 5 \
#--flag gen1_ner0.3_E100_top5 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_top5.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 1 \
#--top_k 10 \
#--flag gen1_ner0.3_E100_top10 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_top10.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 2 \
#--flag gen1_ner0.3_E100_beam2 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_beam2.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 5 \
#--flag gen1_ner0.3_E100_beam5 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_beam5.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/eval_refiner.py \
#--data_type wow \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_test_random_split.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/wow/our_test_random_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/wow/gen1_ner0.3_E100_nops/epoch7-valid_lm_loss2.6884.ckpt \
#--num_beams 10 \
#--flag gen1_ner0.3_E100_beam10 \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow_refiner_nops/ > test_log/wow_gen1_ner0.3_E100_beam10.out &

##########################################################
#################       cmudog      #######################
###########################################################
#CUDA_VISIBLE_DEVICES=2 nohup python ner_model/eval_refiner.py \
#--data_type cmudog \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
#--num_beams 1 \
#--flag gen1_ner0.3_E100_greedy \
#--mode gen_imp \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_greedy.out &

CUDA_VISIBLE_DEVICES=2 nohup python ner_model/eval_refiner.py \
--data_type cmudog \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
--num_beams 1 \
--top_k 5 \
--flag gen1_ner0.3_E100_top5 \
--mode gen_imp \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_top5.out &

CUDA_VISIBLE_DEVICES=2 nohup python ner_model/eval_refiner.py \
--data_type cmudog \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
--num_beams 1 \
--top_k 10 \
--flag gen1_ner0.3_E100_top10 \
--mode gen_imp \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_top10.out &

CUDA_VISIBLE_DEVICES=3 nohup python ner_model/eval_refiner.py \
--data_type cmudog \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
--num_beams 2 \
--flag gen1_ner0.3_E100_beam2 \
--mode gen_imp \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_beam2.out &

CUDA_VISIBLE_DEVICES=3 nohup python ner_model/eval_refiner.py \
--data_type cmudog \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
--num_beams 5 \
--flag gen1_ner0.3_E100_beam5 \
--mode gen_imp \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_beam5.out &

CUDA_VISIBLE_DEVICES=3 nohup python ner_model/eval_refiner.py \
--data_type cmudog \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/cmudog/gen1_ner0.3_E100/epoch21-valid_lm_loss3.3986.ckpt \
--num_beams 10 \
--flag gen1_ner0.3_E100_beam10 \
--mode gen_imp \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/cmudog_refiner/ > test_log/cmudog_gen1_ner0.3_E100_beam10.out &
