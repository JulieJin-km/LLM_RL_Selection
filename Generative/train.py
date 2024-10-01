
from models import SummarizationModule,TranslationModule
from callback import get_early_stopping_callback,Seq2SeqLoggingCallback,get_checkpoint_callback
from lightning_base import generic_train,logger


import argparse

import pytorch_lightning as pl
from pathlib import Path
import torch
import shutil

import copy
import os
import glob
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args,model=None)-> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    SummarizationModule.istest = False
    if model is None:
        if"summarization" in args.task:
            model = SummarizationModule(args)
        else:
            model = TranslationModule(args)

    #################
    if args.ckpt_path is not None:
        model = model.load_from_checkpoint(
            args.ckpt_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            freeze_encoder=args.freeze_encoder,
            max_epochs=args.max_epochs,
            train_batch_size=args.train_batch_size,
            eval_beams=args.eval_beams,
            eval_batch_size=args.eval_batch_size,
            fuse_num=args.fuse_num,
            type_embedding=args.type_embedding,
        )
        if not args.freeze_encoder:
            for par in model.model.get_encoder().parameters():
                par.requires_grad = True
        print('******************************')
        print('Continue training from:', args.ckpt_path)
        # print('Parameters:', model.hparams)
        print('******************************')
    #################

    dataset = Path(args.data_dir).name
    if args.logger_name == "default" or args.fast_dev_run or str(args.output_dir).startswith("/tmp") or str(args.output_dir).startswith("/var"):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger
        project = os.environ.get("WANDB_PROJECT",dataset)
        logger = WandbLogger(name=model.output_dir.name,project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(name=model.output_dir.name,project=f"hf_{dataset}")
        # hf means huggingface

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric,args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric=="loss"
    init_state_dict = copy.deepcopy(model).state_dict()
    args.do_train = True
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    #args.do_train = False
    # reload the trainer
    #for i in range(10)
        #print("{}_begin:{}".format(i,torch.cuda.memory_allocated()))
        #print(i)

        #trainer.fit(model, val_dataloaders=model.val_dataloader())
        #model.load_state_dict(init_state_dict)

    #shutil.rmtree("./models/delete")
    #os.mkdir("./models/delete")

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir,"*.ckpt"),recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)


    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model

if __name__ == "__main__":
    set_seed(1015)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()
    print("Training/evaluation parameters %s", args)
    
    main(args)