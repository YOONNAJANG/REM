CUDA_VISIBLE_DEVICES=4 nohup python train_cmudog.py \
--train_dataset_path="/home/data/leejeongwoo/projects/focus/Refiner/baselines/cmudog/data/Conversations/processed/train.json" \
--dev_dataset_path="/home/data/leejeongwoo/projects/focus/Refiner/baselines/cmudog/data/Conversations/processed/valid.json" \
--output_dir="./output/" \
--flag="epochs_100" \
--n_epochs=100 > cmudog_100epochs.log &