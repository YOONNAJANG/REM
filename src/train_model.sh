focus_train_path=data/FoCus/train.json
focus_train_cache_path=data/FoCus/train_cache.tar.gz
focus_dev_path=data/FoCus/dev.json
focus_dev_cache_path=data/FoCus/dev_cache.tar.gz

wow_train_path=data/WoW/train.json
wow_train_cache_path=data/WoW/train_cache.tar.gz
wow_dev_path=data/WoW/dev.json
wow_dev_cache_path=data/WoW/dev_cache.tar.gz

cmu_train_path=data/CMUDoG/train.json
cmu_train_cache_path=data/CMUDoG/train_cache.tar.gz
cmu_dev_path=data/CMUDoG/dev.json
cmu_dev_cache_path=data/CMUDoG/dev_cache.tar.gz

output_dir=output/


##BART base focus
CUDA_VISIBLE_DEVICES=0 python train_refiner.py --ner_coef 0.3 --epochs 100 --pretrained_model facebook/bart-base --lr 6.25e-5 --data_type focus --output_dir ${output_dir}/bart_base_focus --train_path ${focus_train_path} --train_cache_path ${focus_train_cache_path} --dev_path ${focus_dev_path} --dev_cache_path ${focus_dev_cache_path}
#BART base wow
#CUDA_VISIBLE_DEVICES=0 python src/train_refiner.py --ner_coef 0.7 --epochs 100 --pretrained_model facebook/bart-base --lr 6.25e-5 --data_type wow --output_dir ${output_dir}/bart_base_wow --train_path ${wow_train_path} --train_cache_path ${wow_train_cache_path} --dev_path ${wow_dev_path} --dev_cache_path ${wow_dev_cache_path}
##BART base cmudog
#CUDA_VISIBLE_DEVICES=0 python src/train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model facebook/bart-base --lr 6.25e-5 --data_type cmudog --output_dir ${output_dir}/bart_base_cmudog --train_path ${cmu_train_path} --train_cache_path ${cmu_train_cache_path} --dev_path ${cmu_dev_path} --dev_cache_path ${cmu_dev_cache_path}

###BART large focus
#CUDA_VISIBLE_DEVICES=0 python src/train_refiner.py --ner_coef 0.3 --epochs 100 --pretrained_model facebook/bart-large --data_type focus --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir ${output_dir}/bart_large_focus --train_path ${focus_train_path} --train_cache_path ${focus_train_cache_path} --dev_path ${focus_dev_path} --dev_cache_path ${focus_dev_cache_path}
###BART large wow
#CUDA_VISIBLE_DEVICES=0 python src/train_refiner.py --ner_coef 0.7 --epochs 100 --pretrained_model facebook/bart-large --data_type wow --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir ${output_dir}/bart_large_wow --train_path ${wow_train_path} --train_cache_path ${wow_train_cache_path} --dev_path ${wow_dev_path} --dev_cache_path ${wow_dev_cache_path}
###BART large cmudog
#CUDA_VISIBLE_DEVICES=0 python src/train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model facebook/bart-large --data_type cmudog --train_batch_size 8 --grad_accum 32 --lr 6.25e-5 --output_dir ${output_dir}/bart_large_cmudog --train_path ${cmu_train_path} --train_cache_path ${cmu_train_cache_path} --dev_path ${cmu_dev_path} --dev_cache_path ${cmu_dev_cache_path}

