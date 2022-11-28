from setproctitle import setproctitle
setproctitle("Yoonna")

import os
from argparse import ArgumentParser
import wandb
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model import Model
# from data_utils import make_logdir, get_gpt2_data_loaders, get_t5_data_loaders, get_bart_data_loaders, add_special_tokens, add_special_tokens_gpt2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"



def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="default value for PLMs {facebook/bart-base, t5-base}")
    parser.add_argument("--data_type", type=str, default='revised', help="{original, revised, convai, empchat}")
    parser.add_argument("--checkpoint", type=str, default="", help="load checkpoint and resume train")
    parser.add_argument("--train_dataset_path", type=str, default="/home/yoonna/persona_chat/data/personachat/train_both_original.txt")
    parser.add_argument("--valid_dataset_path", type=str, default="/home/yoonna/persona_chat/data/personachat/valid_both_original.txt")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--grad_accum", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="{AdamW, AdamP}")
    parser.add_argument("--lr_scheduler", type=str, default="lambdalr", help="{exp, lambdalr}")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--flag", type=str, default="both_ori_both_ori", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=644128, help="{644128, 128464}")
    parser.add_argument("--output_dir", type=str, default="/home/mnt/yoonna/personachat/model/bart_base/", help="directory where the model to be saved on")
    parser.add_argument("--test_mode", type=bool, default=False)

    #for p-tuning
    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="3,3,3") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])

    # for manual tuning
    parser.add_argument("--manual_tuning", type=bool, default=False)

    # for few-shot setting
    parser.add_argument("--few_shot_setting", type=bool, default=False)
    parser.add_argument("--few_shot_data", type=str, default="a", choices=["a", "b", "c"])
    parser.add_argument("--few_shot_num", type=int, default=10, choices=[10, 50, 100, 500])





    args = vars(parser.parse_args())

    torch.manual_seed(args['seed'])
    seed_everything(args['seed'], workers=True)


    if args['gpu_num'] == 1:
        args['distributed'] = False
    elif args['gpu_num'] > 1:
        args['distributed'] = True
    else:
        raise NotImplementedError

    model = Model(**args)
    model.to(args['device'])

    if args['flag']:
        flag = args['flag']
    else:
        flag = 'E'+str(args['epochs'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=args['output_dir']+flag,
        filename='epoch{epoch}-ppl{valid_ppl:.4f}',
        monitor='valid_ppl',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor='valid_ppl',
        patience=2,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor()


    wandb.init(project='persona_chat', reinit=True, config=args)
    wandb_logger = WandbLogger(project='persona_chat')
    wandb.watch(model, log_freq=10)


    trainer_args = {
        'callbacks': [checkpoint_callback, early_stopping, lr_monitor],
        'max_epochs': args['epochs'],
        'fast_dev_run': args['test_mode'],
        'num_sanity_val_steps': 2, #None if args['test_mode'] else 0
        'accumulate_grad_batches': args['grad_accum'],
        'gradient_clip_val': args['max_norm'],
        'deterministic': False,
        'gpus': args['gpu_num'],
        'strategy': "deepspeed_stage_2",
        'precision': 32,
        'logger': wandb_logger}


    if args['checkpoint']:
        print(':: Load checkpoint from hparams :')
        print(torch.load(args['checkpoint'])['hyper_parameters'])
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['checkpoint'])

    print(":: Start Training ::")
    trainer = Trainer(**trainer_args)

    trainer.fit(model)
    wandb.finish()

if __name__ == "__main__":
    main()
