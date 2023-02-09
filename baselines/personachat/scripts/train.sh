#CUDA_VISIBLE_DEVICES=4,5 nohup nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--flag both_ori_both_ori > train_log/both_ori_both_ori.log &
#
#
#CUDA_VISIBLE_DEVICES=4,5 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_other_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_other_original.txt \
#--flag other_ori_other_ori > train_log/other_ori_other_ori.log &
#
#
#CUDA_VISIBLE_DEVICES=1,3 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_none_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_none_original.txt \
#--flag none_ori_none_ori > train_log/none_ori_none_ori.log &
#
#CUDA_VISIBLE_DEVICES=6,7 nohup nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--flag both_rev_both_rev > train_log/both_rev_both_rev.log &
#
#
#CUDA_VISIBLE_DEVICES=4,5 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_other_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_other_revised.txt \
#--flag other_rev_other_rev > train_log/other_rev_other_rev.log &
#
#
#CUDA_VISIBLE_DEVICES=1,3 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_self_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_self_revised.txt \
#--flag self_rev_self_rev > train_log/self_rev_self_rev.log &


#########################Ptuning-FULLSHOT###################################

#####Ptuning FULLSHOT with original


#CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 3,3,3 \
#--flag ptuning3_both_ori_both_ori \
#--ptuning True > train_log/ptuning3_both_ori_both_ori.log &


#CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--flag manual_tuning_both_ori_both_ori \
#--manual_tuning True > train_log/manual_both_ori_both_ori.log &

#
#CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 1,1,1 \
#--flag ptuning1_both_ori_both_ori \
#--ptuning True > train_log/ptuning1_both_ori_both_ori.log &


#CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 10,10,10 \
#--flag ptuning10_both_ori_both_ori \
#--ptuning True > train_log/ptuning10_both_ori_both_ori.log &
#
#CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 50,50,50 \
#--flag ptuning50_both_ori_both_ori \
#--ptuning True > train_log/ptuning50_both_ori_both_ori.log &

#CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_original.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_original.txt \
#--template 100,100,100 \
#--flag ptuning100_both_ori_both_ori \
#--ptuning True > train_log/ptuning100_both_ori_both_ori.log &









#####Ptuning FULLSHOT with revised data

#
#CUDA_VISIBLE_DEVICES=6 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 3,3,3 \
#--flag ptuning3_both_rev_both_rev \
#--ptuning True > train_log/ptuning3_both_rev_both_rev.log &
#
#
#CUDA_VISIBLE_DEVICES=7 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 1,1,1 \
#--flag ptuning1_both_rev_both_rev \
#--ptuning True > train_log/ptuning1_both_rev_both_rev.log &


#CUDA_VISIBLE_DEVICES=5 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 10,10,10 \
#--flag ptuning10_both_rev_both_rev \
#--ptuning True > train_log/ptuning10_both_rev_both_rev.log &





#CUDA_VISIBLE_DEVICES=0 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 50,50,50 \
#--flag ptuning50_both_rev_both_rev \
#--ptuning True > train_log/ptuning50_both_rev_both_rev.log &
#
#CUDA_VISIBLE_DEVICES=1 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--template 100,100,100 \
#--flag ptuning100_both_rev_both_rev \
#--ptuning True > train_log/ptuning100_both_rev_both_rev.log &
#
#CUDA_VISIBLE_DEVICES=2 nohup python train.py \
#--train_dataset_path /home/yoonna/persona_chat/data/personachat/train_both_revised.txt \
#--valid_dataset_path /home/yoonna/persona_chat/data/personachat/valid_both_revised.txt \
#--flag manual_tuning_both_rev_both_rev \
#--manual_tuning True > train_log/manual_both_rev_both_rev.log &


##########DONE#############
