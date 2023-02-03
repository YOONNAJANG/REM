CUDA_VISIBLE_DEVICES=4 python ner_model/eval_refiner.py \
--data_type focus \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/test.json \
--test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/focus/ours_cache_test.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/data/ssh5131/focus_modeling/regen_add_ner_v2/focus/gen1_ner0.7_E100_metric/epoch21-valid_lm_loss1.5818.ckpt \
--num_beams 1 \
--flag gen1_ner0.7_E100_metric_greedy \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/focus_refiner/