#bart-large prediction
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_cmu_bart-large_refine > eval_logs/threshold_0.0_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_cmu_bart-large_refine > eval_logs/threshold_0.1_cmu_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_cmu_bart-large_refine > eval_logs/threshold_0.2_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_cmu_bart-large_refine > eval_logs/threshold_0.3_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_cmu_bart-large_refine > eval_logs/threshold_0.4_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_bart-large_refine > eval_logs/threshold_0.5_cmu_bart-large_refine.out &


CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_cmu_bart-large_refine_only.out &



#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_cmu_bart-large_refine > eval_logs/threshold_0.6_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_cmu_bart-large_refine > eval_logs/threshold_0.7_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_cmu_bart-large_refine > eval_logs/threshold_0.8_cmu_bart-large_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_cmu_bart-large_refine > eval_logs/threshold_0.9_cmu_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_cmu_bart-large_refine > eval_logs/threshold_1.0_cmu_bart-large_refine.out &

##woner
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_woner_bart-large_refine > eval_logs/threshold_0.5_cmu_woner_bart-large_refine.out &

#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_woner_bart-large_refine_only --only_score_refine True > eval_logs/threshold_0.5_cmu_woner_bart-large_refine_only.out &



##bart-base prediction

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_cmu_bart-base_refine > eval_logs/threshold_0.0_cmu_bart-base_refine.out &


#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.1 --seed 644128 --flag threshold_0.1_cmu_bart-base_refine > eval_logs/threshold_0.1_cmu_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.2 --seed 644128 --flag threshold_0.2_cmu_bart-base_refine > eval_logs/threshold_0.2_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_cmu_bart-base_refine > eval_logs/threshold_0.3_cmu_bart-base_refine.out &
#
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.4 --seed 644128 --flag threshold_0.4_cmu_bart-base_refine > eval_logs/threshold_0.4_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_bart-base_refine > eval_logs/threshold_0.5_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_cmu_bart-base_refine_only.out &


#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.6 --seed 644128 --flag threshold_0.6_cmu_bart-base_refine > eval_logs/threshold_0.6_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_cmu_bart-base_refine > eval_logs/threshold_0.7_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.8 --seed 644128 --flag threshold_0.8_cmu_bart-base_refine > eval_logs/threshold_0.8_cmu_bart-base_refine.out &
#
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.9 --seed 644128 --flag threshold_0.9_cmu_bart-base_refine > eval_logs/threshold_0.9_cmu_bart-base_refine.out &

#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch18-03.0136.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_cmu_bart-base_refine > eval_logs/threshold_1.0_cmu_bart-base_refine.out &


#bart-base -NER
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/epoch14-03.0137.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_woner_bart-base_refine > eval_logs/threshold_0.5_cmu_woner_bart-base_refine.out &
##

#bart-base -NER
CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-base --mode ner \
--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/our_test_cache.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/epoch14-03.0137.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/cmudog_wo_prompt_100/ \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_woner_bart-base_refine_only --only_score_refine True > eval_logs/threshold_0.5_cmu_woner_bart-base_refine_only.out &
#


#
##### ITDD prediction
#
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.0 --seed 644128 --flag threshold_0.0_cmu_ITDD_refine > eval_logs/threshold_0.0_cmu_ITDD_refine.out &

#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.3 --seed 644128 --flag threshold_0.3_cmu_ITDD_refine > eval_logs/threshold_0.3_cmu_ITDD_refine.out &
#
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_ITDD_refine > eval_logs/threshold_0.5_cmu_ITDD_refine.out &
#
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.7 --seed 644128 --flag threshold_0.7_cmu_ITDD_refine > eval_logs/threshold_0.7_cmu_ITDD_refine.out &
#
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/epoch10-02.8730.ckpt --output_dir /home/data/yoonna/Refiner/output/ner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 1.0 --seed 644128 --flag threshold_1.0_cmu_ITDD_refine > eval_logs/threshold_1.0_cmu_ITDD_refine.out &

##BART-large -NER
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large --mode ner \
#--test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/cmudog/for_ITDD_test_cache.tar.gz \
#--checkpoint /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/epoch6-02.9085.ckpt --output_dir /home/data/yoonna/Refiner/output/woner/bart_large_cmudog_wo_prompt_100/ \
#--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag threshold_0.5_cmu_woner_ITDD_refine > eval_logs/threshold_0.5_cmu_woner_ITDD_refine.out &