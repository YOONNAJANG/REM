
#yoonna
#ner focus
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > logs/wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100 --ptuning True > logs/w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init keyword > logs/w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init entity > logs/w_prompt_entity_100.log &
#ner wow
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type wow --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100 > logs/wow_wo_prompt_100.log &
#ner cmudog
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type cmudog --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100 > logs/cmudog_wo_prompt_100.log &

#T5 ner focus
CUDA_VISIBLE_DEVICES=1 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100 > logs/t5_wo_prompt_100.log &
#ner wow
CUDA_VISIBLE_DEVICES=2 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type wow --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100 > logs/t5_wow_wo_prompt_100.log &
#ner cmudog
CUDA_VISIBLE_DEVICES=3 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type cmudog --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100 > logs/t5_cmudog_wo_prompt_100.log &


##gen
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100 > logs/gen_wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss > logs/gen_wo_prompt_100_wonerloss.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100 --ptuning True > logs/gen_w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/ --ptuning True --target_word_to_init keyword > logs/gen_w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100 --ptuning True --target_word_to_init entity > logs/gen_w_prompt_entity_100.log &

##exp gen wow
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type wow --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wow_wo_prompt_100/epoch7-valid_lm_loss2.7045.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wow_wo_prompt_100 > logs/gen_exp_wow_wo_prompt_100.log &
#exp gen cmudog
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type cmudog --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/cmudog_wo_prompt_100/epoch25-valid_lm_loss3.4120.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/cmudog_wo_prompt_100 > logs/gen_exp_cmudog_wo_prompt_100.log &

#exp gen t5 focus
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/t5_wo_prompt_100/epoch69-valid_lm_loss2.0213.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/t5_wo_prompt_100 > logs/gen_exp_t5_wo_prompt_100.log &
##exp gen wow
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type wow --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/t5_wow_wo_prompt_100/epoch71-valid_lm_loss3.3868.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/t5_wow_wo_prompt_100 > logs/gen_exp_t5_wow_wo_prompt_100.log &
#exp gen cmudog
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --pretrained_model t5-small --data_type cmudog --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/t5_cmudog_wo_prompt_100/epoch99-valid_lm_loss5.0774.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/t5_cmudog_wo_prompt_100 > logs/gen_exp_t5_cmudog_wo_prompt_100.log &


#fewshot 100
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot --fewshot True --fewnum 100 > logs/gen_wo_prompt_1_100shot.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.0 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_100shot_wonerloss --fewshot True --fewnum 100 > logs/gen_wo_prompt_1_100shot_wonerloss.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_100shot --ptuning True --fewshot True --fewnum 100 > logs/gen_w_prompt_rand_1_100shot.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_keyword_1_100shot.log &
#CUDA_VISIBLE_DEVICES=3 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_100shot --ptuning True --fewshot True --fewnum 100 --target_word_to_init keyword > logs/gen_w_prompt_entity_1_100shot.log &

#fewshot 500
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot --fewshot True --fewnum 500 > logs/gen_wo_prompt_1_500shot.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.0 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_1_500shot_wonerloss --fewshot True --fewnum 500 > logs/gen_wo_prompt_1_500shot_wonerloss.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_1_500shot --ptuning True --fewshot True --fewnum 500 > logs/gen_w_prompt_rand_1_500shot.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_1_500shot --ptuning True --fewshot True --fewnum 500 --target_word_to_init keyword > logs/gen_w_prompt_keyword_1_500shot.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 1 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_1_500shot --ptuning True --fewshot True --fewnum 500 --target_word_to_init keyword > logs/gen_w_prompt_entity_1_500shot.log &


#for test

#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 1.0 --epochs 100 --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_ner1.0 > logs/gen_imp_wo_prompt_100_ner1.0.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > logs/wo_prompt_100.log &

#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100 > logs/gen_imp_wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_wonerloss > logs/gen_imp_wo_prompt_100_wonerloss.log &

#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss > logs/gen_wo_prompt_100_wonerloss.log &


