#FoCus
CUDA_VISIBLE_DEVICES=4 python src/eval_refiner.py --data_type focus --pretrained_model facebook/bart-large \
--test_dataset_path data/FoCus/test.json \
--test_dataset_cache data/FoCus/test_cache.tar.gz \
--checkpoint {ckpt_path} --output_dir {output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag focus_bart-large_refine


#CMUDoG
CUDA_VISIBLE_DEVICES=4 python src/eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large \
--test_dataset_path data/CMUDoG/test.json \
--test_dataset_cache data/CMUDoG/test_cache.tar.gz \
--checkpoint {ckpt_path} --output_dir {output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag cmu_bart-large_refine


#WoW
CUDA_VISIBLE_DEVICES=3 python src/eval_refiner.py --data_type wow --pretrained_model facebook/bart-large \
--test_dataset_path data/WoW/test.json \
--test_dataset_cache data/WoW/test_cache.tar.gz \
--checkpoint {ckpt_path} --output_dir {output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 wow_bart-large_refine
