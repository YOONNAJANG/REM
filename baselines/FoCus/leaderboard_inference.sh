#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.


#Baseline official test
#CUDA_VISIBLE_DEVICES=7 nohup python inference_official_test.py \
#--model_name BART \
#--model_path facebook/bart-base \
#--checkpoint /home/mnt/yoonna/focus_modeling/model/FoCus_baseline_BART/epoch1-ppl5.8532.ckpt \
#--retrieval_type TFIDF \
#--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
#--temperature 0.7 \
#--top_p 0.9 \
#--leaderboard \
#--flag FoCus_baseline_BART_official_test > official_test_log/FoCus_baseline_BART_official_test.log &

#Baseline workshop test
CUDA_VISIBLE_DEVICES=7 nohup python inference_official_test.py \
--test_dataset_path /home/mnt/yoonna/focus_modeling/data/focus_workshop_public.json \
--test_dataset_cache /home/mnt/yoonna/focus_modeling/data/workshop_test_cache.tar.gz \
--model_name BART \
--model_path facebook/bart-base \
--checkpoint /home/mnt/yoonna/focus_modeling/model/FoCus_baseline_BART/epoch1-ppl5.8532.ckpt \
--retrieval_type TFIDF \
--landmark_dic /home/mnt/yoonna/focus_modeling/our_data/all_landmark_dic.json \
--temperature 0.7 \
--top_p 0.9 \
--leaderboard \
--flag FoCus_baseline_BART > official_test_log/FoCus_baseline_BART_workshop.log &

#test code for T5, baseline
#Ms 문의 메일 - 박사 integrated? 해당 분야 아니어도 지원 가능?



#test LED
#CUDA_VISIBLE_DEVICES=1 nohup python evaluate_test_prev.py --flag led_test --model_name LED --model_path allenai/led-base-16384 --checkpoint /home/mnt/yoonna/FoCus_modeling/model/led_test/epoch0-ppl0.0000.ckpt > test_log/test_led.log &

echo
