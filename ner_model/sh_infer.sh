###########################################################
##################       FoCus      #######################
###########################################################
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_greedy.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_greedy_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_greedy.out &

#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_top5.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_top5_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_top5.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_top10.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_top10_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_top10.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_beam2.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_beam2_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_beam2.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_beam5.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_beam5_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_beam5.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/focus_chatgpt_refine_beam10.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--flag focus_chatgpt_refine_beam10_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/focus/ > test_log/focus_chatgpt_refine_beam10.out &

##########################################################
#################       WoW      #######################
###########################################################
#CUDA_VISIBLE_DEVICES=1 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_greedy.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_greedy_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_greedy.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_top5.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_top5_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_top5.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_top10.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_top10_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_top10.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_beam2.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_beam2_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_beam2.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_beam5.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_beam5_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_beam5.out &
#
#CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
#--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/wow_chatgpt_refine_beam10.json \
#--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--flag wow_chatgpt_refine_beam10_w_kblue \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/wow/ > test_log/wow_chatgpt_refine_beam10.out &


##########################################################
#################       cmudog      #######################
###########################################################
CUDA_VISIBLE_DEVICES=1 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_greedy.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_greedy_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_greedy.out &

CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_top5.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_top5_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_top5.out &

CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_top10.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_top10_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_top10.out &

CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_beam2.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_beam2_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_beam2.out &

CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_beam5.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_beam5_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_beam5.out &

CUDA_VISIBLE_DEVICES=0 nohup python ner_model/only_kblue.py \
--test_dataset_path /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/cmudog_chatgpt_refine_beam10.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
--flag cmudog_chatgpt_refine_beam10_w_kblue \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/chatgpt/cmudog/ > test_log/cmudog_chatgpt_refine_beam10.out &