import sys
from os import path

from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import EarlyStopping  # , ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_mem_model.lightning_model import CorefModel
from callbacks.model_checkpoint import MyModelCheckpoint


def experiment(args):
    seed_everything(args.seed)

    # Tensorboard logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        version=args.model_name,
        name=None,
    )
    # Callbacks
    lr_logger = LearningRateLogger()

    checkpoint_callback = MyModelCheckpoint(
        verbose=True,
        monitor='fscore',
        mode='max',
        save_top_k=1,
        period=-1,
        save_last=True,
        prefix='coref_')
    early_stop_callback = EarlyStopping(
        monitor='fscore',
        patience=10,
        verbose=True,
        mode='max'
    )

    # Resume from checkpoint automatically
    resume_from_checkpoint = None
    potential_old_checkpoint = path.join(logger.log_dir, 'checkpoints/coref_last.ckpt')
    if path.isfile(potential_old_checkpoint):
        resume_from_checkpoint = potential_old_checkpoint
        print("Resuming training from: ", potential_old_checkpoint)
    sys.stdout.flush()
    args.max_epochs = min(10, args.max_epochs)

    trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        precision=32,
        weights_save_path=args.save_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
        gradient_clip_val=1.0, terminate_on_nan=True,
        row_log_interval=10, log_save_interval=10)

    coref_model = CorefModel(**vars(args))
    trainer.fit(coref_model)

    print("Best validation model path: ", checkpoint_callback.best_model_path)
    print("Best validation performance:", checkpoint_callback.best_model_score)
    #
    # trainer.test(ckpt_path='best')
    # test_res = trainer.test()

    best_score = checkpoint_callback.best_model_score
    try:
        val_perf = best_score.item()
    except AttributeError:
        val_perf = best_score

    return val_perf
