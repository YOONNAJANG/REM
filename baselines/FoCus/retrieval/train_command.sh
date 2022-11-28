# To train the retrieval model
CUDA_VISIBLE_DEVICES=2 python train_knowledge_retrieval.py \
--epochs=1 \
--batch_size=1 \
--learning_rate=3e-5 \
--gradient_accumulation_steps=256