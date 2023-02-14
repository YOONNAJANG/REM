###full shots###
#wo_prompt_100
CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_100_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100/ --flag output_b5 --num_beams 5 > eval_logs/wo_prompt_100_b5.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100/ --flag output_b1 > eval_logs/wo_prompt_100_b1.log &

#w_prompt_rand_100
CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_100_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_100_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/epoch5-valid_lm_loss2.3420.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_100_b1.log &

#w_prompt_keyword_100
CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_100_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_100_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/epoch14-valid_lm_loss1.9141.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_100_b1.log &

#w_prompt_entity_100
CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_100_b10.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_100_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/epoch8-valid_lm_loss2.0607.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_100/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_100_b1.log &


###100 shots###


#wo_prompt_1_100shot
CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/ --flag output_b10 --num_beams 10 > eval_logs/wo_prompt_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/ --flag output_b5 --num_beams 5 > eval_logs/wo_prompt_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/epoch0-valid_lm_loss1.8686.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_100shot/ --flag output_b1 > eval_logs/wo_prompt_1_100shot_b1.log &

#w_prompt_rand_1_100shot
CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/epoch0-valid_lm_loss2.9618.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_1_100shot_b1.log &

#w_prompt_keyword_1_100shot
CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/epoch0-valid_lm_loss3.0615.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_1_100shot_b1.log &

#w_prompt_entity_1_100shot
CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b10.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/epoch0-valid_lm_loss3.1553.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_100shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_1_100shot_b1.log &


##500 shots###

#wo_prompt_1_500shot
CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/ --flag output_b10 --num_beams 10 > eval_logs/w_prompt_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=1 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/ --flag output_b5 --num_beams 5 > eval_logs/w_prompt_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/epoch0-valid_lm_loss1.6767.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_1_500shot/ --flag output_b1 > eval_logs/w_prompt_1_500shot_b1.log &

#w_prompt_rand_1_500shot
CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=2 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=3 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/epoch0-valid_lm_loss2.9167.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_rand_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_rand_1_500shot_b1.log &

#w_prompt_keyword_1_500shot
CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=4 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/epoch0-valid_lm_loss3.0033.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_keyword_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_keyword_1_500shot_b1.log &

#w_prompt_entity_1_500shot
CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/ --flag output_b10 --num_beams 10 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b10.log &
#CUDA_VISIBLE_DEVICES=0 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/ --flag output_b5 --num_beams 5 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b5.log &
#CUDA_VISIBLE_DEVICES=7 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/epoch0-valid_lm_loss3.0715.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/w_prompt_entity_1_500shot/ --flag output_b1 --ptuning True > eval_logs/w_prompt_entity_1_500shot_b1.log &



#done

