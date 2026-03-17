import os
from modules import LightningBERTForRegression
import torch
from torch import nn
import lightning.pytorch as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# set up logging
logger = logging.getLogger(__name__)
# login to wandb
wandb.login()


@hydra.main(version_base=None, config_path=".", config_name="hydra_config")
def main(cfg: DictConfig) -> None:
    # print the configuration
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    # wandb tracking variables
    os.environ["WANDB_PROJECT"] = cfg.loggers.wandb.wandb_project
    os.environ["WANDB_ENTITY"] = cfg.loggers.wandb.wandb_entity
    os.environ["WANDB_USERNAME"] = cfg.loggers.wandb.wandb_username
    os.environ["WANDB_USER_EMAIL"] = cfg.loggers.wandb.wandb_user_email
    os.environ["WANDB_NAME"] = str(cfg.loggers.wandb.wandb_name)
    os.environ["WANDB_JOB_NAME"] = cfg.loggers.wandb.wandb_job_name
    os.environ["WANDB_JOB_TYPE"] = cfg.loggers.wandb.wandb_job_type
    os.environ["WANDB_TAGS"] = cfg.loggers.wandb.wandb_tags
    os.environ["WANDB_RUN_GROUP"] = cfg.loggers.wandb.wandb_run_group
    os.environ["WANDB_DIR"] = cfg.loggers.wandb.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = cfg.loggers.wandb.wandb_cache_dir
    os.environ["WANDB_DATA_DIR"] = cfg.loggers.wandb.wandb_data_dir
    os.environ["WANDB_ARTIFACT_DIR"] = cfg.loggers.wandb.wandb_artifact_dir
    os.environ["WANDB_API_KEY"] = cfg.loggers.wandb.wandb_api_key
    os.environ["WANDB_WATCH"] = cfg.loggers.wandb.wandb_watch
    local_rank = int(torch.cuda.current_device())
    resume = None
    id = str(cfg.loggers.wandb.wandb_name)
    if cfg.loggers.wandb.wandb_resume:
        # make sure the same run ID is used and processed otherwise give an error
        # https://docs.wandb.ai/guides/runs/resuming/
        resume = "must"
        id = os.environ["WANDB_RUN_ID"] = cfg.loggers.wandb.wandb_run_id
    if local_rank == 0:
        run = wandb.init(
            settings=wandb.Settings(init_timeout=1200),
            entity=cfg.loggers.wandb.wandb_entity,
            project=cfg.loggers.wandb.wandb_project,
            group=cfg.loggers.wandb.wandb_run_group,
            tags=cfg.loggers.wandb.wandb_tags.split(","),
            resume=resume,
            id=id,
        )
    # create the default root dir if it does not exist
    if not os.path.exists(cfg.dirs.default_root_dir):
        os.makedirs(cfg.dirs.default_root_dir, exist_ok=True)

    # seed everything
    L.seed_everything(cfg.trainer.seed, workers=True)

    # define the loss functions
    loss_fn = nn.MSELoss()

    # initialize the lightning model
    lightning_model = LightningBERTForRegression(
        pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        dataset_path=cfg.data.dataset_path,
        test_size=cfg.data.test_size,
        seed=cfg.trainer.seed,
        learning_rate=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        optimizer=None,
        lr_scheduler=None,
        loss_fn=loss_fn,
        stack_depth=cfg.model.stack_depth,
        p=cfg.model.p,
        label_column=cfg.data.label_column,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # initialize the logger
    logger_name = cfg.loggers.logger
    logger_obj = None
    if logger_name is None or logger_name == "tensorboard":
        logger_obj = TensorBoardLogger(
            save_dir=cfg.loggers.tensorboard.save_dir,
            name=cfg.loggers.tensorboard.name,
            sub_dir=cfg.loggers.tensorboard.sub_dir,
        )
    elif logger_name == "wandb":
        logger_obj = WandbLogger(
            name=cfg.loggers.wandb.wandb_name,
            save_dir=cfg.loggers.wandb.wandb_dir,
            offline=cfg.loggers.wandb.wandb_offline,
            log_model=cfg.loggers.wandb.wandb_log_model,
            prefix=(
                ""
                if cfg.loggers.wandb.wandb_prefix is None
                else cfg.loggers.wandb.wandb_prefix
            ),
            settings=wandb.Settings(init_timeout=1200),
            entity=cfg.loggers.wandb.wandb_entity,
            project=cfg.loggers.wandb.wandb_project,
            group=cfg.loggers.wandb.wandb_run_group,
            tags=cfg.loggers.wandb.wandb_tags.split(","),
            resume=resume,
            id=id,
        )
        logger_obj.watch(
            lightning_model,
            log=cfg.loggers.wandb.wandb_watch,
            log_freq=cfg.loggers.wandb.wandb_log_freq,
        )
    # setup callbacks
    callbacks = []
    if cfg.callbacks.get("early_stopping", False):
        # early stopping callback
        early_stop_callback = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            patience=cfg.callbacks.early_stopping.patience,
            verbose=cfg.callbacks.early_stopping.verbose,
            mode=cfg.callbacks.early_stopping.mode,
        )
        callbacks.append(early_stop_callback)

    if cfg.callbacks:
        # setup the learning rate monitor callback
        lr_monitor = LearningRateMonitor(
            logging_interval=cfg.callbacks.lr_monitor.logging_interval,
            log_momentum=cfg.callbacks.lr_monitor.log_momentum,
            log_weight_decay=cfg.callbacks.lr_monitor.log_weight_decay,
        )
        callbacks.append(lr_monitor)

    # initialize the trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.num_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        default_root_dir=cfg.dirs.default_root_dir,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        precision=cfg.trainer.precision,
        logger=logger_obj,
    )

    # train the model
    # NOTE: the dataloaders are created inside the LightningModule
    # using native hooks of PyTorch Lightning
    trainer.fit(lightning_model)

    # test the model
    # NOTE: need a separate trainer for testing
    # when using DDP strategy
    if str(cfg.trainer.strategy).lower() == "ddp":
        tester = L.Trainer(
            max_epochs=cfg.trainer.num_epochs,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            num_nodes=cfg.trainer.num_nodes,
            strategy=cfg.trainer.strategy,
            default_root_dir=cfg.dirs.default_root_dir,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            callbacks=[early_stop_callback],
            precision=cfg.trainer.precision,
            deterministic=cfg.trainer.deterministic,
            logger=logger_obj,
        )
        tester.test(lightning_model)
    else:
        trainer.test(lightning_model)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_USERNAME",
        "WANDB_USER_EMAIL",
        "WANDB_NAME",
        "WANDB_JOB_NAME",
        "WANDB_JOB_TYPE",
        "WANDB_RUN_GROUP",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
        "WANDB_ARTIFACT_DIR",
        "WANDB_WATCH",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


if __name__ == "__main__":
    # reset the wandb environment variables
    reset_wandb_env()
    # call the main function
    main()
