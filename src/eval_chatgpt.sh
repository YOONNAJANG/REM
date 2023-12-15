

#####CHATGPT#####
#focus chatgpt
###before_refine
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_large_focus_chatgpt_before_refine > eval_logs/bart_large_focus_chatgpt_before_refine.out &

###bart-large
###b10
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch6-01.2499.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_chatgpt_ner_refine_b10 > eval_logs/bart_large_focus_chatgpt_ner_refine_b10.out &
#
###b1
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch6-01.2499.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_chatgpt_ner_refine_b1 > eval_logs/bart_large_focus_chatgpt_ner_refine_b1.out &

##b5
CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_focus_chatgpt_ner_refine_b5 > eval_logs/bart_large_focus_chatgpt_ner_refine_b5.out &


###BART-base b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch10-01.2185.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_focus_chatgpt_ner_refine_b5 > eval_logs/bart_base_focus_chatgpt_ner_refine_b5.out &


###wow chatgpt
##bart-large
##before refine
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/chatgpt_pretty_test_random_split_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_large_wow_chatgpt_ner_before_refine > eval_logs/bart_large_wow_chatgpt_ner_before_refine.out &
#
##b10
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/chatgpt_pretty_test_random_split_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_chatgpt_ner_refine_b10 > eval_logs/bart_large_wow_chatgpt_ner_refine_b10.out &
#
##b5
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/chatgpt_pretty_test_random_split_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_wow_chatgpt_ner_refine_b5 > eval_logs/bart_large_wow_chatgpt_ner_refine_b5.out &

##BART-base b5
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/chatgpt_pretty_test_random_split_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/wow/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch9-02.4309.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_wow_chatgpt_ner_refine_b5 > eval_logs/bart_base_wow_chatgpt_ner_refine_b5.out &

##cmudog chatgpt
#bart-large
###before_refine
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/chatgpt_result_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-03.2327.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 0.0 --seed 644128 --flag bart_large_cmu_chatgpt_ner_before_refine > eval_logs/bart_large_cmu_chatgpt_ner_before_refine.out &

##b10
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/chatgpt_result_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 10 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_chatgpt_ner_refine_b10 > eval_logs/bart_large_cmu_chatgpt_ner_refine_b10.out &
#
##b1
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/chatgpt_result_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 1 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_chatgpt_ner_refine_b1 > eval_logs/bart_large_cmu_chatgpt_ner_refine_b1.out &
#
###b5
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-large --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/chatgpt_result_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_large_cmu_chatgpt_ner_refine_b5 > eval_logs/bart_large_cmu_chatgpt_ner_refine_b5.out &

##BART-base b5
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type chatgpt --pretrained_model facebook/bart-base --mode ner --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/chatgpt_result_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/cmudog/ours_cache_test.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag bart_base_cmu_chatgpt_ner_refine_b5 > eval_logs/bart_base_cmu_chatgpt_ner_refine_b5.out &
