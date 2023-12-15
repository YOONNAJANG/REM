

#####CHATGPT#####
#wow chatgpt

###before
#CUDA_VISIBLE_DEVICES=7 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_refine_with_llm_without_entity.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag wow_before_refine > eval_logs/wow_llm_before_refine.out &

#####woner
#CUDA_VISIBLE_DEVICES=0 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_refine_with_llm_without_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/threshold_0.5_wow_bart-base_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag wow_woner_threshold05 > eval_logs/wow_woner_threshold05.out &
#
#
####with ner
#CUDA_VISIBLE_DEVICES=2 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/baselines/wow/output/2023_emnlp/bart_base/test_random_split_beam5_09k_B_B_refine_with_llm_with_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/threshold_0.5_wow_bart-base_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag wow_withner_threshold05 > eval_logs/wow_withner_threshold05.out &



#CMUDoG chatgpt
####before
#CUDA_VISIBLE_DEVICES=2 nohup python eval_llm_output.py --data_type cmudog \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/cmudog_test_refine_with_llm_without_entity.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag cmudog_before_refine > eval_logs/cmudog_llm_before_refine.out &



####woner
#CUDA_VISIBLE_DEVICES=3 nohup python eval_llm_output.py --data_type cmudog \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/cmudog_test_refine_with_llm_without_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/threshold_0.5_cmu_bart-large_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag cmudog_woner_threshold05 > eval_logs/cmudog_woner_threshold05.out &
#
####with ner
#CUDA_VISIBLE_DEVICES=4 nohup python eval_llm_output.py --data_type cmudog \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/cmudog_test_refine_with_llm_with_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/threshold_0.5_cmu_bart-large_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag cmudog_withner_threshold05 > eval_logs/cmudog_withner_threshold05.out &


##FoCus chatgpt

#

####before
#CUDA_VISIBLE_DEVICES=1 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/focus_test_beam5_09k_B_B_refine_with_llm_without_entity.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag focus_before_refine > eval_logs/focus_llm_before_refine.out &

####woner
#CUDA_VISIBLE_DEVICES=0 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/focus_test_beam5_09k_B_B_refine_with_llm_without_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/threshold_0.5_focus_bart-large_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag focus_woner_threshold05 > eval_logs/focus_woner_threshold05.out &
#
####with ner
#CUDA_VISIBLE_DEVICES=2 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/focus_test_beam5_09k_B_B_refine_with_llm_with_entity.json \
#--threshold_dataset_path /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/threshold_0.5_focus_bart-large_refine.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag focus_withner_threshold05 > eval_logs/focus_withner_threshold05.out &


###FoCus chatgpt LLM -> LLM
#
####before
#CUDA_VISIBLE_DEVICES=4 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/focus_w_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag focus_before_refine_LLM2LLM > eval_logs/focus_llm_before_refine_LLM2LLM.out &
#
####woner
#CUDA_VISIBLE_DEVICES=5 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/focus_wo_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag focus_woner_threshold05_LLM2LLM > eval_logs/focus_woner_threshold05_LLM2LLM.out &
#
####with ner
#CUDA_VISIBLE_DEVICES=6 nohup python eval_llm_output.py --data_type focus \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/focus_w_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag focus_withner_threshold05_LLM2LLM > eval_logs/focus_withner_threshold05_LLM2LLM.out &

##WoW chatgpt LLM -> LLM

####before
#CUDA_VISIBLE_DEVICES=4 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/wow_wo_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag wow_before_refine_LLM2LLM > eval_logs/wow_llm_before_refine_LLM2LLM.out &
#
####woner
#CUDA_VISIBLE_DEVICES=5 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/wow_wo_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag wow_woner_threshold05_LLM2LLM > eval_logs/wow_woner_threshold05_LLM2LLM.out &
#
####with ner
#CUDA_VISIBLE_DEVICES=6 nohup python eval_llm_output.py --data_type wow \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/wow_w_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag wow_withner_threshold05_LLM2LLM > eval_logs/wow_withner_threshold05_LLM2LLM.out &
#

###CMU chatgpt LLM -> LLM
#
####before
#CUDA_VISIBLE_DEVICES=4 nohup python eval_llm_output.py --data_type cmu \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/cmudog_wo_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --before_refine True --flag cmu_before_refine_LLM2LLM > eval_logs/cmu_llm_before_refine_LLM2LLM.out &
#
####woner
#CUDA_VISIBLE_DEVICES=5 nohup python eval_llm_output.py --data_type cmu \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/cmudog_wo_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag cmu_woner_threshold05_LLM2LLM > eval_logs/cmu_woner_threshold05_LLM2LLM.out &
#
####with ner
#CUDA_VISIBLE_DEVICES=6 nohup python eval_llm_output.py --data_type cmu \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/cmudog_w_entity_eval_format.json \
#--output_dir /home/data/yoonna/Refiner/output/llm/ \
#--seed 644128 --flag cmu_withner_threshold05_LLM2LLM > eval_logs/cmu_withner_threshold05_LLM2LLM.out &



