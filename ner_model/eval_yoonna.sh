###full shots###
#wo_prompt_100
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/ --flag output_b5 --num_beams 5 > eval_logs/wo_prompt_100_b5.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100/ --flag output_b1 > eval_logs/wo_prompt_100_b1.log &

#woner
#RUNNING
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss/epoch5-valid_lm_loss1.4742.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_wonerloss_b10.log &

#wo_prompt_100_imp
#RUNNING
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100/epoch3-valid_lm_loss1.4935.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_imp_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_ner1.0/epoch5-valid_lm_loss1.4884.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_ner1.0/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_imp_ner1.0_b10.log &

#woner
#RUNNING
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_wonerloss/epoch5-valid_lm_loss1.4856.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_wonerloss/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_imp_wonerloss_b10.log &


#w_prompt_rand_100
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_100_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_100_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_100_b1.log &

#w_prompt_keyword_100
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_100_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_100_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_100_b1.log &

#w_prompt_entity_100
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_100_b10.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_100_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_100_b1.log &



#####CHATGPT#####
#gen_exp
CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type chatgpt --mode gen_exp --test_dataset_path /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/chatgpt_002_test_ours_for_inference.json --test_dataset_cache /home/data/ssh5131/focus_modeling/for_refiner_v2/chatgpt/focus/ours_cache_test.tar.gz \
--checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss/epoch5-valid_lm_loss1.4742.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss/ \
--num_beams 10 --flag focus_chatgpt_gen_exp_refine_b10 > eval_logs/focus_chatgpt_refine_b10.out &


###100 shots###


#wo_prompt_1_100shot
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/ --flag output_b5 --num_beams 5 > eval_logs/wo_prompt_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot/ --flag output_b1 > eval_logs/wo_prompt_1_100shot_b1.log &
#woner
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot_wonerloss/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot_wonerloss/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_1_100shot_wonerloss_b10.log &


#w_prompt_rand_1_100shot
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b1.log &

#w_prompt_keyword_1_100shot
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b1.log &

#w_prompt_entity_1_100shot
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b1.log &


##500 shots###

#wo_prompt_1_500shot
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/ --flag output_b5 --num_beams 5 > eval_logs/wo_prompt_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot/ --flag output_b1 > eval_logs/wo_prompt_1_500shot_b1.log &
#woner
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot_wonerloss/epoch0-valid_lm_loss1.6762.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot_wonerloss/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_1_500shot_wonerloss_b10.log &


#w_prompt_rand_1_500shot
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b1.log &

#w_prompt_keyword_1_500shot
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b1.log &

#w_prompt_entity_1_500shot
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b1.log &








#done

