###FoCus chatgpt LLM -> BARTREM
#
####before
#CUDA_VISIBLE_DEVICES=3 nohup python eval_llm_output_with_plm.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/focus_w_entity_eval_format.json \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt \
#--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--seed 644128 --num_beams 5 --refine_threshold 0.0 --flag focus_before_refine_LLM2BART > eval_logs/focus_llm_before_refine_LLM2BART.out &
#
#####BART-large rem 0.5
#CUDA_VISIBLE_DEVICES=4 nohup python eval_llm_output_with_plm.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/focus_w_entity_eval_format.json \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.0976.ckpt \
#--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--seed 644128 --num_beams 5 --refine_threshold 0.5 --flag focus_before_refine_LLM2BART > eval_logs/focus_llm_after_refine_LLM2BART.out &

#
#####CMUDoG chatgpt LLM -> BARTREM
#
####before
#CUDA_VISIBLE_DEVICES=5 nohup python eval_llm_output_with_plm.py --data_type cmu --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/cmudog_w_entity_eval_format.json \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt \
#--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--seed 644128 --num_beams 5 --refine_threshold 0.0 --flag cmu_before_refine_LLM2BART > eval_logs/cmu_llm_before_refine_LLM2BART.out &
#
####BART-large rem 0.5
CUDA_VISIBLE_DEVICES=6 nohup python eval_llm_output_with_plm.py --data_type cmu --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/cmudog_w_entity_eval_format.json \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt \
--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
--seed 644128 --num_beams 5 --refine_threshold 0.5 --flag cmu_after_refine_LLM2BART > eval_logs/cmu_llm_after_refine_LLM2BART.out &

###WoW chatgpt LLM -> BARTREM
#
####before
#CUDA_VISIBLE_DEVICES=0 nohup python eval_llm_output_with_plm.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/wow_wo_entity_eval_format.json \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt \
#--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--seed 644128 --num_beams 5 --refine_threshold 0.0 --flag wow_before_refine_LLM2BART > eval_logs/wow_llm_before_refine_LLM2BART.out &
#
#####BART-large rem 0.5
#CUDA_VISIBLE_DEVICES=1 nohup python eval_llm_output_with_plm.py --data_type wow --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/leejeongwoo/projects/focus/Refiner/refine_with_llm/llm_gen_to_llm_refine/wow_wo_entity_eval_format.json \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/epoch5-02.2709.ckpt \
#--output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wow_wo_prompt_100/ \
#--seed 644128 --num_beams 5 --refine_threshold 0.5 --flag wow_after_refine_LLM2BART > eval_logs/wow_llm_after_refine_LLM2BART.out &