import os
import time
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Optional, Union
from datasets import Dataset, load_from_disk
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForMaskedLM,
    default_data_collator,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from numpy.typing import ArrayLike
import lightning as L

# from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# import wandb
import logging

logger = logging.getLogger(__name__)


def _load_model(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    revision: Optional[str] = "main",
    cache_dir: str = "./tmp",
    token: Optional[Union[str, bool]] = False,
    trust_remote_code: bool = True,
) -> transformers.BertForMaskedLM:
    # load the model configuration
    # https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/auto#transformers.AutoConfig.from_pretrained
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    # load the model
    # https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    model = BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        config=config,
        cache_dir=cache_dir,
        token=token,
    )
    return model


def load_tokenizer(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    revision: Optional[str] = "main",
    cache_dir: str = "./tmp",
    token: Optional[Union[str, bool]] = False,
    trust_remote_code: bool = True,
) -> transformers.AutoTokenizer:
    # load the tokenizer
    # https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


class LightningBERTForRegression(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        dataset_path: str = None,
        test_size: float = 0.2,
        seed: int = 1234,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Union[nn.Module, dict]] = None,
        stack_depth: int = 1,
        p: float = 0.5,
        label_column: str = "label",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super(LightningBERTForRegression, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if loss_fn is None:
            logger.warning(
                "No loss function provided. Using Mean Squared Error Loss by default."
            )
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        self.stack_depth = stack_depth
        self.p = p
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        # data placeholders
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        # save hyperparameters
        self.save_hyperparameters(ignore=["loss_fn"])

    def _prepare_data(self):
        # columns to select
        use_columns = ["input_ids", "attention_mask"]
        label_column = self.label_column
        use_columns.append(label_column)

        # Load dataset (download on disk if not already present)
        hub_ds = (
            load_from_disk(self.dataset_path)
            .select_columns(use_columns)
            .shuffle(self.seed)
            .take(1_000_000)  # limit to 1 million samples
        )

        # set the dataset format to PyTorch tensors
        hub_ds.set_format(type="torch")

        # create train/valid/test split (80/10/10)
        train_temp_split = hub_ds.train_test_split(
            test_size=self.test_size, seed=self.seed, shuffle=True
        )
        valid_test_split = train_temp_split["test"].train_test_split(
            test_size=0.5, seed=self.seed, shuffle=True
        )
        self.train_data = train_temp_split["train"]
        self.valid_data = valid_test_split["train"]
        self.test_data = valid_test_split["test"]
        # # clean up memory
        # del hub_ds
        # del train_temp_split
        # del valid_test_split

    def _prepare_model(self):
        # load the pretrained BERT model
        if self.pretrained_model_name_or_path:
            model = _load_model(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path
            )
        else:
            raise ValueError("pretrained_model_name_or_path must be provided")
        self.base_model = model.base_model
        self.hidden_size = self.base_model.config.hidden_size
        # define a new regression head
        layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(p=self.p),
            ]
            * self.stack_depth
        )
        layers.append(nn.Linear(self.hidden_size, 1))
        self.regression_head = nn.Sequential(*layers)

    def setup(self, stage: Optional[str] = None):
        self._prepare_data()
        self._prepare_model()

    def train_dataloader(self):
        if self.train_data is None:
            raise ValueError("train_data must be provided")
        # create the training dataloader
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.valid_data is None:
            return None
        # create the validation dataloader
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        # create the test dataloader
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def forward(self, input_ids, attention_mask):
        # pass the input through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # get the pooled output
        pooled_output = outputs.last_hidden_state[:, 0, :]
        # pass the pooled output through the regression head
        pooled_output = self.regression_head(pooled_output)
        # return outputs
        return pooled_output

    def _compute_loss(self, predictions, targets, prefix=""):
        if isinstance(self.loss_fn, dict):
            return (
                {
                    prefix + name: fxn(predictions, targets)
                    for name, fxn in self.loss_fn.items()
                },
            )
        else:
            return {f"{prefix}loss": self.loss_fn(predictions, targets)}

    def _compute_metrics(self, predictions, targets, prefix=""):
        metrics = dict()
        metrics[f"{prefix}mse"] = mean_squared_error(predictions, targets, squared=True)
        metrics[f"{prefix}rmse"] = mean_squared_error(
            predictions, targets, squared=False
        )
        metrics[f"{prefix}r2"] = r2_score(predictions, targets)
        metrics[f"{prefix}mae"] = mean_absolute_error(predictions, targets)
        return metrics

    def _run_step(self, batch):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        targets = batch[self.label_column].float()
        predictions = self(**inputs).squeeze(-1)
        return predictions, targets

    def training_step(self, batch, batch_idx):
        predictions, targets = self._run_step(batch)
        metrics_dict = self._compute_metrics(predictions, targets, prefix="train/")
        metrics_dict.update(
            self._compute_loss(predictions, targets, prefix="")
        )  # the loss key for training must be "loss"
        self.log_dict(
            dictionary=metrics_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            reduce_fx="mean",
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )
        return metrics_dict

    def validation_step(self, batch, batch_idx):
        predictions, targets = self._run_step(batch)
        metrics_dict = self._compute_metrics(predictions, targets, prefix="val/")
        metrics_dict.update(self._compute_loss(predictions, targets, prefix="val/"))
        self.log_dict(
            dictionary=metrics_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            reduce_fx="mean",
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )
        return metrics_dict

    def test_step(self, batch, batch_idx):
        predictions, targets = self._run_step(batch)
        metrics_dict = self._compute_metrics(predictions, targets, prefix="test/")
        metrics_dict.update(self._compute_loss(predictions, targets, prefix="test/"))
        self.log_dict(
            dictionary=metrics_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            reduce_fx="mean",
            sync_dist=True if torch.cuda.device_count() > 1 else False,
        )
        return metrics_dict

    def configure_optimizers(self):
        if self.optimizer is not None and self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        else:
            self.optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=1,
                min_lr=self.learning_rate * 1e-3,
                eps=1e-8,
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "monitor": "val/loss",
            }
