CUDA_VISIBLE_DEVICES=7 python infer_cmudog_dev.py \
--checkpoint="./output/epochs_100/epoch21-ppl16.5576.ckpt" \
--train_dataset_path="./data/Conversations/processed/train.json" \
--dev_dataset_path="./data/Conversations/processed/test.json" \
--flag="epochs_100_test" \
--output_dir="./output/epochs_100/"
