CUDA_VISIBLE_DEVICES=7 python ner_model/eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/regen_add_ner/test100/epoch5-valid_lm_loss1.4686.ckpt --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/test/ --flag beam5output --num_beams 5
CUDA_VISIBLE_DEVICES=7 python ner_model/eval_refiner.py --data_type focus --checkpoint /home/data/yoonna/Refiner/output/regen_add_ner/test100/epoch5-valid_lm_loss1.4686.ckpt --output_dir /home/data/yoonna/Refiner/output/regen_add_ner/test/ --flag beam1output