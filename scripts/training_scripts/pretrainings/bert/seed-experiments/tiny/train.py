###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: March 2025                                                            #
# Description:                                                                #
# Script for pre-training Bert masked language model on canonical isomeric    #
# SMILES from the PubChem (version 04-18-2024) dataset                        #
# This script requires a hydra configuration YAML file (hydra_config.yaml)    #
# to run.                                                                     #
###############################################################################
# import the necessary modules
import os
import math
import hydra
from omegaconf import OmegaConf, DictConfig
import evaluate
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_from_disk
import wandb
import logging
from datetime import datetime

# set the logger
logger = logging.getLogger(__name__)
wandb.login()


@hydra.main(version_base=None, config_path=".", config_name="hydra_config")
def main(cfg: DictConfig) -> None:
    # print the configuration
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    # wandb tracking variables
    os.environ["WANDB_PROJECT"] = cfg.wandb.wandb_project
    os.environ["WANDB_ENTITY"] = cfg.wandb.wandb_entity
    os.environ["WANDB_USERNAME"] = cfg.wandb.wandb_username
    os.environ["WANDB_USER_EMAIL"] = cfg.wandb.wandb_user_email
    os.environ["WANDB_NAME"] = str(cfg.wandb.wandb_name)
    os.environ["WANDB_JOB_NAME"] = cfg.wandb.wandb_job_name
    os.environ["WANDB_JOB_TYPE"] = cfg.wandb.wandb_job_type
    os.environ["WANDB_TAGS"] = cfg.wandb.wandb_tags
    os.environ["WANDB_RUN_GROUP"] = cfg.wandb.wandb_run_group
    os.environ["WANDB_DIR"] = cfg.wandb.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = cfg.wandb.wandb_cache_dir
    os.environ["WANDB_DATA_DIR"] = cfg.wandb.wandb_data_dir
    os.environ["WANDB_ARTIFACT_DIR"] = cfg.wandb.wandb_artifact_dir
    os.environ["WANDB_API_KEY"] = cfg.wandb.wandb_api_key
    os.environ["WANDB_WATCH"] = cfg.wandb.wandb_watch
    os.environ["WANDB_LOG_MODEL"] = cfg.wandb.wandb_log_model
    # get the local rank, global rank, group rank, local world size, role world size, world size, and process id
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    # goes from 0 to max_nnodes. For single group per node, this is the node rank
    group_rank = int(os.environ["GROUP_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    process_id = int(os.getpid())
    print(20 * "=")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"GLOBAL_RANK: {global_rank}")
    print(f"GROUP_RANK: {group_rank}")
    print(f"LOCAL_WORLD_SIZE: {local_world_size}")
    print(f"ROLE_WORLD_SIZE: {role_world_size}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"PID: {process_id}")
    print(20 * "=")
    resume = None
    id = (
        str(cfg.wandb.wandb_name)
        + "-"
        + str(global_rank)
        + "-"
        + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    if cfg.wandb.wandb_resume:
        # make sure the same run ID is used and processed otherwise give an error
        # https://docs.wandb.ai/guides/runs/resuming/
        resume = "must"
        id = os.environ["WANDB_RUN_ID"] = cfg.wandb.wandb_run_id
    if local_rank == 0:
        run = wandb.init(
            entity=cfg.wandb.wandb_entity,
            project=cfg.wandb.wandb_project,
            group=cfg.wandb.wandb_run_group,
            tags=cfg.wandb.wandb_tags.split(","),
            resume=resume,
            id=id,
        )
    # instantiate the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.processing_class,
        revision="main",
        cache_dir=cfg.dirs.cache_dir,
        token=cfg.model.token,
        trust_remote_code=True,
    )
    # instantiate the BERT Base configuration
    bert_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=cfg.model.hidden_size,
        hidden_act=cfg.model.hidden_act,
        initializer_range=cfg.model.initializer_range,
        hidden_dropout_prob=cfg.model.hidden_dropout_prob,
        num_attention_heads=cfg.model.num_attention_heads,
        type_vocab_size=cfg.model.type_vocab_size,
        max_position_embeddings=cfg.model.max_position_embeddings,
        num_hidden_layers=cfg.model.num_hidden_layers,
        intermediate_size=cfg.model.intermediate_size,
        attention_probs_dropout_prob=cfg.model.attention_probs_dropout_prob,
    )

    # instantiate the BERT Base Masked Language bert_mlm
    bert_mlm = BertForMaskedLM(bert_config)

    # load the tokenized dataset
    if not os.path.exists(cfg.data.dataset_path):
        raise ValueError(
            "The tokenized PubChem dataset does not exist. Please create it first."
        )
    tokenized_pubchem_ds_split = load_from_disk(cfg.data.dataset_path).remove_columns(
        [
            "smiles",
        ]
    )

    tokenized_pubchem_ds_split.set_format(type="torch")

    # split the data into train and validation sets with 80% and 20% ratio
    tokenized_pubchem_ds_split = tokenized_pubchem_ds_split.train_test_split(
        test_size=0.2, seed=cfg.trainer.data_seed, shuffle=True
    )

    # instantiate the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.trainer.mlm_probability
    )

    # instantiate the metric(s) for evaluation in distributed setting with shared file system
    # without concatenating the results/logits for each batch
    if cfg.trainer.batch_eval_metrics:
        accuracy = evaluate.load(
            "accuracy",
            cache_dir=cfg.dirs.cache_dir,
            num_process=1,
            process_id=0,
            experiment_id=global_rank,
        )
        f1 = evaluate.load(
            "f1",
            cache_dir=cfg.dirs.cache_dir,
            num_process=1,
            process_id=0,
            experiment_id=global_rank,
        )
    else:
        accuracy = evaluate.load("accuracy", cache_dir=cfg.dirs.cache_dir)
        f1 = evaluate.load("f1", cache_dir=cfg.dirs.cache_dir)

    # define the preprocess_logits_for_metrics function
    def preprocess_logits_for_metrics(logits, labels):
        # labels will be None if the data does not contain labels
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # define the compute_metrics function
    def compute_metrics(eval_preds, compute_result=None):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return {
            "eval_accuracy": accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "eval_macro_f1": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
            "eval_weighted_f1": f1.compute(
                predictions=preds, references=labels, average="weighted"
            )["f1"],
        }

    # set the parameters
    if (
        not os.path.exists(cfg.trainer.output_dir)
        or cfg.trainer.overwrite_output_dir is True
    ):
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)
    if not os.path.exists(cfg.trainer.log_dir):
        os.makedirs(cfg.trainer.log_dir, exist_ok=True)

    # define the training arguments
    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir,
        overwrite_output_dir=cfg.trainer.overwrite_output_dir,
        resume_from_checkpoint=cfg.trainer.resume_from_checkpoint,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        num_train_epochs=cfg.trainer.num_train_epochs,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        logging_dir=cfg.trainer.log_dir,
        eval_strategy=cfg.trainer.eval_strategy,
        logging_strategy=cfg.trainer.logging_strategy,
        save_strategy=cfg.trainer.save_strategy,
        eval_steps=(
            cfg.trainer.eval_steps if cfg.trainer.eval_strategy == "steps" else None
        ),
        logging_steps=(
            cfg.trainer.logging_steps
            if cfg.trainer.logging_strategy == "steps"
            else None
        ),
        save_steps=(
            cfg.trainer.save_steps if cfg.trainer.save_strategy == "steps" else None
        ),
        save_safetensors=True,
        seed=cfg.trainer.seed,
        data_seed=cfg.trainer.data_seed,
        fp16=cfg.trainer.fp16,
        ddp_backend=cfg.trainer.ddp_backend,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.trainer.dataloader_prefetch_factor,
        dataloader_persistent_workers=cfg.trainer.dataloader_persistent_workers,
        dataloader_pin_memory=cfg.trainer.dataloader_pin_memory,
        gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        eval_accumulation_steps=(
            cfg.trainer.eval_accumulation_steps
            if isinstance(cfg.trainer.eval_accumulation_steps, int)
            else None
        ),
        batch_eval_metrics=cfg.trainer.batch_eval_metrics,
        torch_empty_cache_steps=cfg.trainer.torch_empty_cache_steps,
        run_name=str(cfg.trainer.run_name),
        remove_unused_columns=True,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        greater_is_better=cfg.trainer.greater_is_better,
        report_to=["tensorboard", "wandb", "mlflow"],
        push_to_hub=cfg.trainer.push_to_hub,
        hub_private_repo=True,
        hub_token=cfg.model.token,
        hub_strategy="all_checkpoints",
        hub_always_push=False,
    )

    # instantiate the Trainer class
    trainer = Trainer(
        model=bert_mlm,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_pubchem_ds_split["train"],
        eval_dataset=tokenized_pubchem_ds_split["test"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.trainer.early_stopping_patience,
                early_stopping_threshold=cfg.trainer.early_stopping_threshold,
            ),
        ],
    )

    # train the model
    train_results = trainer.train(
        resume_from_checkpoint=cfg.trainer.resume_from_checkpoint
    )

    # save the model and the state
    trainer.save_model()
    trainer.save_state()

    # log the results
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # evaluation
    metrics = trainer.evaluate(eval_dataset=tokenized_pubchem_ds_split["test"])
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["eval_perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if local_rank == 0:
        # finish the wandb run
        run.finish()


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
        "WANDB_LOG_MODEL",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


if __name__ == "__main__":
    # reset the wandb environment variables
    reset_wandb_env()
    # call the main function
    main()
