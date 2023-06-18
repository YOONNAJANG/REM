#bart-base prediction


#bart-large prediction


#INFO prediction
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_focus_INFO_refine > eval_logs/threshold_0.0_focus_INFO_refine.out &

CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_focus_INFO_refine > eval_logs/threshold_0.1_focus_INFO_refine.out &

CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_focus_INFO_refine > eval_logs/threshold_0.2_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_focus_INFO_refine > eval_logs/threshold_0.3_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_focus_INFO_refine > eval_logs/threshold_0.5_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_focus_INFO_refine > eval_logs/threshold_0.7_focus_INFO_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner.json \
#--test_dataset_cache /home/data/yoonna/Refiner/ner_model/inf_qualitative_results_0_ner_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/epoch8-01.2570.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_focus_INFO_refine > eval_logs/threshold_1.0_focus_INFO_refine.out &