CUDA_VISIBLE_DEVICES=4 nohup python wow/train_wow.py --model_name BART --model_path facebook/bart-base --n_epochs 100 \
 --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
 --dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/valid_random_split.json \
 --flag train_wow_origin_v2 \
 --output_dir /home/data/ssh5131/focus_modeling/model/ > wow_train_v2.out &