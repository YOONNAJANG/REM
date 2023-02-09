CUDA_VISIBLE_DEVICES=0 python train_cmudog.py \
--train_dataset_path="./data/Conversations/processed/train.json" \
--dev_dataset_path="./data/Conversations/processed/valid.json" \
--output_dir="./output/" \
--flag="epochs_100" \
--n_epochs=100