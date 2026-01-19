#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
"""

import argparse
import json
import logging
import math
import os
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from peft import PeftModel, prepare_model_for_kbit_training


logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--validation_split_percentage", default=5)

    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_keep_linebreaks", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")

    # custom peft args
    parser.add_argument("--lora_adapter_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--evaluate_every_n_steps", type=int, default=None)

    parser.add_argument(
        "--freeze_alpha_default",
        type=float,
        default=0.1,
        help="Default grad scaling alpha for frozen LoRA ranks when no per-rank alpha is provided.",
    )
    parser.add_argument(
        "--freeze_alpha_dict_filename",
        type=str,
        default="freeze_alpha_dict.json",
        help="Optional JSON filename under lora_adapter_dir to provide per-rank alpha via s_vals.",
    )

    args = parser.parse_args()

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")

    if args.train_file is not None:
        ext = args.train_file.split(".")[-1]
        if ext not in ["csv", "json", "txt"]:
            raise ValueError("`train_file` should be a csv, json or txt file.")
    if args.validation_file is not None:
        ext = args.validation_file.split(".")[-1]
        if ext not in ["csv", "json", "txt"]:
            raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub and args.output_dir is None:
        raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    if args.lora_adapter_dir is not None and args.lora_adapter_dir == "NA":
        args.lora_adapter_dir = None

    return args


def _compute_alpha_vec(
    matched_key: str,
    freeze_r: int,
    freeze_alpha_dict: dict | None,
    default_freeze_alpha: float,
):
    if freeze_r <= 0:
        return [], "none", None

    alpha = 5.0

    if freeze_alpha_dict is not None and matched_key in freeze_alpha_dict:
        entry = freeze_alpha_dict[matched_key]
        s_vals = None
        if isinstance(entry, dict) and "s_vals" in entry:
            s_vals = entry["s_vals"]
        elif isinstance(entry, list):
            s_vals = entry

        if isinstance(s_vals, list) and len(s_vals) > freeze_r:
            s_ref = float(s_vals[freeze_r])

            max_sigma = max(float(s) for s in s_vals)
            if max_sigma <= 1e-30:
                return [default_freeze_alpha] * freeze_r, "file", s_ref

            alpha_vec = []
            for i in range(freeze_r):
                sigma_i = float(s_vals[i])
                denom = alpha * sigma_i + max_sigma
                if denom <= 1e-30:
                    a_i = default_freeze_alpha
                else:
                    lambda_i = ((alpha + 1.0) * sigma_i) / denom
                    a_i = 1.0 - lambda_i
                    if a_i < 0.0:
                        a_i = 0.0
                    elif a_i > 1.0:
                        a_i = 1.0
                alpha_vec.append(a_i)

            return alpha_vec, "file", s_ref

    return [default_freeze_alpha] * freeze_r, "default", None


def main():
    args = parse_args()

    mixed_precision = "bf16"
    accelerator_log_kwargs = {"mixed_precision": mixed_precision}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )
    IS_MAIN = accelerator.is_main_process

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    api = None
    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # dataset
    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            trust_remote_code=args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        extension = None
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # config/tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError("Need --tokenizer_name if training from scratch without --model_name_or_path.")

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    _model_loaded_from_local = bool(args.model_name_or_path and Path(args.model_name_or_path).exists())

    _model_is_bnb_quantized = False
    if hasattr(model, "is_quantized") and model.is_quantized:
        _model_is_bnb_quantized = True
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    _adapter_is_applied = False
    if args.lora_adapter_dir is not None:
        assert Path(args.lora_adapter_dir).exists(), f"{args.lora_adapter_dir} does not exist."
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir, is_trainable=True)
        _adapter_is_applied = True

    logger.info(f"🔍 model loaded from local: {_model_loaded_from_local}")
    logger.info(f"🔍 model is bnb quantized: {_model_is_bnb_quantized}")
    logger.info(f"🔍 adapter is applied: {_adapter_is_applied}")

    if _adapter_is_applied:
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(
            f"🔍 trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        logger.info(
            f"🔍 Full-finetune: all params: {model.num_parameters()}, trainable: {model.num_parameters(only_trainable=True)}"
        )

    # resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # tokenize & group
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            block_size = min(1024, config.max_position_embeddings)
            logger.warning(
                f"Tokenizer model_max_length too large; using block_size={block_size}."
            )
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"block_size {args.block_size} > tokenizer.model_max_length {tokenizer.model_max_length}; clipping."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"] if "test" in lm_datasets.keys() else None

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    # ---- load freeze dicts ----
    freeze_rank_dict = {}
    freeze_alpha_dict = None
    if args.lora_adapter_dir is not None:
        fr_path = Path(args.lora_adapter_dir) / "freeze_rank_dict.json"
        if fr_path.exists():
            with open(fr_path, "r") as f:
                freeze_rank_dict = json.load(f)

        fa_path = Path(args.lora_adapter_dir) / args.freeze_alpha_dict_filename
        if fa_path.exists():
            with open(fa_path, "r") as f:
                freeze_alpha_dict = json.load(f)

    # ---- LoRA split ----
    if len(freeze_rank_dict) > 0:
        for name, module in model.named_modules():
            adapter_name = "default"

            matched_key = None
            for key in freeze_rank_dict:
                if key in name:
                    matched_key = key
                    break
            if matched_key is None:
                continue

            freeze_r = int(freeze_rank_dict[matched_key])
            if freeze_r <= 0:
                continue

            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]
            total_r = lora_A.out_features
            assert freeze_r < total_r, f"freeze_r ({freeze_r}) must be < LoRA rank ({total_r})"

            old_A = lora_A.weight.data.clone()
            old_B = lora_B.weight.data.clone()

            del module.lora_A
            del module.lora_B

            module.lora_A_frozen = torch.nn.Linear(module.in_features, freeze_r, bias=False)
            module.lora_B_frozen = torch.nn.Linear(freeze_r, module.out_features, bias=False)
            module.lora_A_frozen.weight = torch.nn.Parameter(old_A[:freeze_r], requires_grad=True)
            module.lora_B_frozen.weight = torch.nn.Parameter(old_B[:, :freeze_r], requires_grad=True)

            train_r = total_r - freeze_r
            module.lora_A_train = torch.nn.Linear(module.in_features, train_r, bias=False)
            module.lora_B_train = torch.nn.Linear(train_r, module.out_features, bias=False)
            module.lora_A_train.weight = torch.nn.Parameter(old_A[freeze_r:], requires_grad=True)
            module.lora_B_train.weight = torch.nn.Parameter(old_B[:, freeze_r:], requires_grad=True)

            def lora_forward_split(self, x):
                result = torch.nn.functional.linear(x, self.weight, self.bias)

                active = getattr(self, "active_adapter", "default")
                if isinstance(active, (list, tuple)) and len(active) > 0:
                    an = active[0]
                elif isinstance(active, str):
                    an = active
                else:
                    an = "default"

                if isinstance(getattr(self, "lora_dropout", None), dict) and an in self.lora_dropout:
                    x_dropped = self.lora_dropout[an](x)
                else:
                    x_dropped = x

                if isinstance(getattr(self, "scaling", None), dict) and an in self.scaling:
                    sc = self.scaling[an]
                else:
                    sc = getattr(self, "scaling", 1.0)

                delta_frozen = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, self.lora_A_frozen.weight),
                    self.lora_B_frozen.weight,
                )
                delta_train = torch.nn.functional.linear(
                    torch.nn.functional.linear(x_dropped, self.lora_A_train.weight),
                    self.lora_B_train.weight,
                )
                return result + (delta_frozen + delta_train) * sc

            module.forward = lora_forward_split.__get__(module, module.__class__)

    # ---- Grad scaling hooks ----
    alpha_cache = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        matched_key = None
        for key in freeze_rank_dict:
            if key in n:
                matched_key = key
                break
        if matched_key is None:
            continue

        if ("lora_A_frozen.weight" not in n) and ("lora_B_frozen.weight" not in n):
            continue

        freeze_r = int(freeze_rank_dict[matched_key])

        if matched_key not in alpha_cache:
            alpha_vec, _, _ = _compute_alpha_vec(
                matched_key=matched_key,
                freeze_r=freeze_r,
                freeze_alpha_dict=freeze_alpha_dict,
                default_freeze_alpha=args.freeze_alpha_default,
            )
            alpha_cache[matched_key] = (freeze_r, alpha_vec)

        freeze_r, alpha_vec = alpha_cache[matched_key]

        def make_hook(is_A: bool, alpha_vec_local: list[float]):
            def _hook(grad: torch.Tensor):
                if grad is None:
                    return grad
                a = torch.tensor(alpha_vec_local, device=grad.device, dtype=grad.dtype)
                if is_A:
                    return grad * a.view(-1, 1)
                return grad * a.view(1, -1)

            return _hook

        if "lora_A_frozen.weight" in n:
            p.register_hook(make_hook(True, alpha_vec))
        elif "lora_B_frozen.weight" in n:
            p.register_hook(make_hook(False, alpha_vec))

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    params_with_weight_decay = []
    params_without_weight_decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            params_without_weight_decay.append(p)
        else:
            params_with_weight_decay.append(p)

    optimizer_grouped_parameters = [
        {"params": params_with_weight_decay, "weight_decay": args.weight_decay},
        {"params": params_without_weight_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
        ),
    )

    # Prepare
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    if test_dataloader is not None:
        test_dataloader = accelerator.prepare(test_dataloader)

    if accelerator.distributed_type == DistributedType.XLA:
        model.tie_weights()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Trackers
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        tracker_init_kwargs = {"wandb": {}}
        if args.run_name is not None:
            tracker_init_kwargs["wandb"]["name"] = args.run_name
        if args.wandb_tags is not None:
            tracker_init_kwargs["wandb"]["tags"] = args.wandb_tags
        accelerator.init_trackers("train_clm", experiment_config, tracker_init_kwargs)

    # Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(checkpoint_path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        # eval before training (first epoch only)
        if epoch == starting_epoch:
            model.eval()
            losses = []
            for _, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            logger.info(f"Before training: eval_perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log({"eval_perplexity": perplexity, "eval_loss": eval_loss}, step=completed_steps)

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += float(loss.detach().float().item())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.with_tracking:
                    accelerator.log({"train_step_loss": loss.detach().float()}, step=completed_steps)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.with_tracking:
            accelerator.log({"train_loss": total_loss / max(1, len(train_dataloader))}, step=completed_steps)

        if args.evaluate_every_n_steps is None and completed_steps >= args.max_train_steps:
            model.eval()
            losses = []
            for _, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            logger.info(f"epoch {epoch}: eval_perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log({"eval_perplexity": perplexity, "eval_loss": eval_loss}, step=completed_steps)

        if test_dataloader is not None:
            model.eval()
            losses = []
            for _, batch in enumerate(test_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            try:
                test_perplexity = math.exp(eval_loss)
            except OverflowError:
                test_perplexity = float("inf")

            if args.with_tracking:
                accelerator.log({"test_perplexity": test_perplexity}, step=completed_steps)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.with_tracking:
            accelerator.log({"epoch": epoch}, step=completed_steps)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": float(perplexity)}, f)


if __name__ == "__main__":
    main()
