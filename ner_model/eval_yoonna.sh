#CUDA_VISIBLE_DEVICES=6 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100/ --flag output_b5 --num_beams 5 > logs/wo_prompt_100_b5.log &
#CUDA_VISIBLE_DEVICES=5 nohup python eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/gen/wo_prompt_100/epoch6-valid_lm_loss1.4768.ckpt --output_dir /home/data/yoonna/Refiner/output/gen/wo_prompt_100/ --flag output_b1 > logs/wo_prompt_100_b1.log &

#done

