CUDA_VISIBLE_DEVICES=6 nohup python infer_cmudog_dev.py \
--checkpoint="./output/epochs_100/epoch6-ppl20.7657.ckpt" \
--train_dataset_path="/home/data/leejeongwoo/projects/focus/Refiner/baselines/cmudog/data/Conversations/processed/train.json" \
--dev_dataset_path="/home/data/leejeongwoo/projects/focus/Refiner/baselines/cmudog/data/Conversations/processed/valid.json" \
--flag="epochs_100_test" \
--output_dir="./output/epochs_100/" > cmudog_100epochs_infer.log &
