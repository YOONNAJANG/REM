#CUDA_VISIBLE_DEVICES=0 nohup  python wow/infer_wow_dev.py --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
#--dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json --flag train_output_beam1_09k \
#--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/ --checkpoint /home/data/ssh5131/focus_modeling/model/train_wow_origin_v2/epoch8-ppl20.2096.ckpt > wow/train_infer.out &

CUDA_VISIBLE_DEVICES=0 nohup  python wow/infer_wow_dev.py --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
--dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/valid_random_split.json --flag valid_random_split_output_beam1_09k \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/ --checkpoint /home/data/ssh5131/focus_modeling/model/train_wow_origin_v2/epoch8-ppl20.2096.ckpt > wow/valid_random_split_infer.out &

CUDA_VISIBLE_DEVICES=0 nohup  python wow/infer_wow_dev.py --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
--dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/valid_topic_split.json --flag valid_topic_split_output_beam1_09k \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/ --checkpoint /home/data/ssh5131/focus_modeling/model/train_wow_origin_v2/epoch8-ppl20.2096.ckpt > wow/valid_topic_split_infer.out &


CUDA_VISIBLE_DEVICES=1 nohup python wow/infer_wow_dev.py --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
--dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/test_random_split.json --flag test_random_split_output_beam1_09k \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/ --checkpoint /home/data/ssh5131/focus_modeling/model/train_wow_origin_v2/epoch8-ppl20.2096.ckpt > wow/test_random_split_infer.out &

CUDA_VISIBLE_DEVICES=1 nohup  python wow/infer_wow_dev.py --train_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/train.json \
--dev_dataset_path /home/data/ssh5131/focus_modeling/others/wizard_of_wikipedia/original/test_topic_split.json --flag test_topic_split_output_beam1_09k \
--output_dir /home/data/ssh5131/focus_modeling/eval_output/wow/output/new_v2/  --checkpoint /home/data/ssh5131/focus_modeling/model/train_wow_origin_v2/epoch8-ppl20.2096.ckpt > wow/test_topic_split_infer.out &

