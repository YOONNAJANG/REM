import os
from argparse import ArgumentParser
import wandb
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from model import Model_Eval

# from data_utils import make_logdir, get_gpt2_data_loaders, get_t5_data_loaders, get_bart_data_loaders, add_special_tokens, add_special_tokens_gpt2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"



def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BART", help="{BART, T5}")
    parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="default value for PLMs")
    parser.add_argument("--checkpoint", type=str, default="/home/mnt/yoonna/personachat/model/bart_base/both_ori_both_ori/epoch8-ppl10.7132.ckpt/global_step1155/mp_rank_00_model_states.pt", help="load checkpoint and resume train")
    parser.add_argument("--test_dataset_path", type=str, default="/home/yoonna/persona_chat/data/personachat/test_both_revised.txt")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--grad_accum", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--flag", type=str, default="bart_base_", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=644128)
    parser.add_argument("--output_dir", type=str, default="./inference_ourput/", help="directory where the model to be saved on")
    parser.add_argument("--test_mode", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=1, help="{1, 2, 5, 10}, 1 for greedy decoding")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=40, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum length of the output utterances")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=50, help="Filter top-k tokens before sampling {5, 10}, default=50")

    parser.add_argument("--ptuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="3,3,3") #prompt size
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--vocab_strategy", type=str, default="original", choices=['original', 'shared', 'lama'])


    parser.add_argument("--manual_tuning", type=bool, default=False)


    args = vars(parser.parse_args())

    torch.manual_seed(args['seed'])
    seed_everything(args['seed'], workers=True)
    from setproctitle import setproctitle
    setproctitle("leejeongwoo")

    if args['gpu_num'] == 1:
        args['distributed'] = False
    elif args['gpu_num'] > 1:
        args['distributed'] = True
    else:
        raise NotImplementedError

    model = Model_Eval(**args)
    model.to(args['device'])

    # if args['flag']:
    #     flag = args['flag']
    # else:
    #     flag = 'E'+str(args['epochs'])
    #
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args['output_dir']+flag,
    #     filename='epoch{epoch}-ppl{valid_ppl:.4f}',
    #     monitor='valid_ppl',
    #     save_top_k=3,
    #     mode='min',
    #     auto_insert_metric_name=False,
    # )

    wandb.init(project='persona_chat', reinit=True, config=args)
    wandb_logger = WandbLogger(project='persona_chat')
    wandb.watch(model, log_freq=10)


    trainer_args = {
        # 'callbacks': [checkpoint_callback],
        'max_epochs': args['epochs'],
        'fast_dev_run': args['test_mode'],
        'num_sanity_val_steps': 2, #None if args['test_mode'] else 0
        'accumulate_grad_batches': args['grad_accum'],
        'gradient_clip_val': args['max_norm'],
        'deterministic': False,
        'gpus': args['gpu_num'],
        'strategy': "ddp",
        'precision': 32,
        'logger': wandb_logger}


    if args['checkpoint']:
        print(':: Load checkpoint from hparams :')
        print(torch.load(args['checkpoint'])['hyper_parameters'])
        trainer_args['resume_from_checkpoint'] = os.path.join('checkpoints', args['checkpoint'])

    print(":: Start Testing ::")
    trainer = Trainer(**trainer_args)

    model.freeze()
    with torch.no_grad():
        trainer.test(model)
    wandb.finish()

if __name__ == "__main__":
    main()
