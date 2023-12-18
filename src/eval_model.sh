focus_test_path=../data/FoCus/test.json
focus_test_cache_path=../data/FoCus/test_cache.tar.gz
wow_test_path=../data/WoW/test.json
wow_test_cache_path=../data/WoW/test_cache.tar.gz
cmu_test_path=../data/CMUDoG/test.json
cmu_test_cache_path=../data/CMUDoG/test_cache.tar.gz
ckpt_path=../ckpt/ #model checkpoint path
output_path=../output/

#FoCus
CUDA_VISIBLE_DEVICES=0 python src/eval_refiner.py --data_type focus --pretrained_model facebook/bart-large \
--test_dataset_path ${focus_test_path} \
--test_dataset_cache ${focus_test_cache_path} \
--checkpoint ${ckpt_path} --output_dir ${output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag focus_bart-large_refine


#CMUDoG
CUDA_VISIBLE_DEVICES=1 python src/eval_refiner.py --data_type cmudog --pretrained_model facebook/bart-large \
--test_dataset_path ${cmu_test_path} \
--test_dataset_cache ${cmu_test_cache_path} \
--checkpoint ${ckpt_path} --output_dir ${output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 --flag cmu_bart-large_refine


#WoW
CUDA_VISIBLE_DEVICES=2 python src/eval_refiner.py --data_type wow --pretrained_model facebook/bart-large \
--test_dataset_path ${wow_test_path} \
--test_dataset_cache ${wow_test_cache_path} \
--checkpoint ${ckpt_path} --output_dir ${output_path} \
--num_beams 5 --refine_threshold 0.5 --seed 644128 wow_bart-large_refine
