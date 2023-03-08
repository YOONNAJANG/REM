
#yoonna
#ner
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/wo_prompt_100 > logs/wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100 --ptuning True > logs/w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=5 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init keyword > logs/w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode ner --output_dir /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100 --ptuning True --target_word_to_init entity > logs/w_prompt_entity_100.log &

#gen
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100 > logs/gen_wo_prompt_100.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss > logs/gen_wo_prompt_100_wonerloss.log &
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_rand_100/epoch13-valid_lm_loss2.0143.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_rand_100 --ptuning True > logs/gen_w_prompt_rand_100.log &
#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_keyword_100/epoch22-valid_lm_loss1.9821.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_keyword_100/ --ptuning True --target_word_to_init keyword > logs/gen_w_prompt_keyword_100.log &
#CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/w_prompt_entity_100/epoch9-valid_lm_loss2.0814.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/w_prompt_entity_100 --ptuning True --target_word_to_init entity > logs/gen_w_prompt_entity_100.log &

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

#RUNNING
#CUDA_VISIBLE_DEVICES=6 nohup python train_refiner.py --ner_coef 0.5 --epochs 100 --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100 > logs/gen_imp_wo_prompt_100.log &
CUDA_VISIBLE_DEVICES=4 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_imp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_imp/wo_prompt_100_wonerloss > logs/gen_imp_wo_prompt_100_wonerloss.log &

#CUDA_VISIBLE_DEVICES=7 nohup python train_refiner.py --ner_coef 0.0 --epochs 100 --data_type focus --mode gen_exp --checkpoint /home/data/yoonna/Refiner/output/ner/wo_prompt_100/epoch15-valid_lm_loss1.4855.ckpt --output_dir /home/data/yoonna/Refiner/output/gen_exp/wo_prompt_100_wonerloss > logs/gen_wo_prompt_100_wonerloss.log &


