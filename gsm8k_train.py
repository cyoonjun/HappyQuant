# coding=utf-8
# Copyright 2021 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0

"""
gsm8k_train.py

"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)

from peft import PeftModel, prepare_model_for_kbit_training

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on GSM8K with PEFT LoRA + freeze-rank split + grad scaling hooks")

    # Dataset / IO
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="HF dataset name (default: gsm8k)")
    parser.add_argument("--dataset_config", type=str, default="main", help="HF dataset config (default: main)")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--train_split", type=str, default="train", help="Train split name")
    parser.add_argument("--eval_split", type=str, default="test", help="Eval split name")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--trust_remote_code", type=bool, default=False)

    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)

    # Output / hub
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=str, default=None,
                        help="Save states every N steps or 'epoch'.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    # Tracking
    parser.add_argument("--with_tracking", action="store_true", help="Enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="all",
                        help='Supported platforms are "tensorboard", "wandb", "comet_ml", "clearml". Use "all" to report to all.')
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None)

    # PEFT adapter
    parser.add_argument("--lora_adapter_dir", type=str, default=None, help="'NA' for full-finetune")

    # Freeze alpha 
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

    if args.lora_adapter_dir is not None and args.lora_adapter_dir == "NA":
        args.lora_adapter_dir = None

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an --output_dir to create a repo when --push_to_hub is passed."

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


def _format_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _build_sft_features(tokenizer, question: str, answer: str, max_length: int):
    prompt = _format_gsm8k_prompt(question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_text = " " + answer.strip()  
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids + ([eos] if eos is not None else [])
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    labels = [-100] * min(len(prompt_ids), len(input_ids))
    ans_start = len(labels)
    labels += input_ids[ans_start:]
    labels = labels[:len(input_ids)]

    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def main():
    args = parse_args()

    accelerator = (
        Accelerator(
            log_with=args.report_to,
            project_dir=args.output_dir,
            mixed_precision="bf16",
        )
        if args.with_tracking
        else Accelerator(mixed_precision="fp16")
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

            with open(os.path.join(args.output_dir, ".gitignore"), "w+", encoding="utf-8") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    # Dataset
    raw = load_dataset(args.dataset_name, args.dataset_config)
    train_raw = raw[args.train_split]
    eval_raw = raw[args.eval_split]

    # Model / tokenizer / config
    if args.config_name is not None:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )

    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    _model_loaded_from_local = Path(args.model_name_or_path).exists()

    _model_is_bnb_quantized = False
    if hasattr(model, "is_quantized") and model.is_quantized:
        _model_is_bnb_quantized = True
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
    else:
        if args.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    _adapter_is_applied = False
    if args.lora_adapter_dir is not None:
        assert Path(args.lora_adapter_dir).exists(), f"{args.lora_adapter_dir} does not exist."
        model = PeftModel.from_pretrained(
            model,
            args.lora_adapter_dir,
            is_trainable=True,
            ignore_mismatched_sizes=True,
        )
        _adapter_is_applied = True

    logger.info(f"🔍 model loaded from local: {_model_loaded_from_local}")
    logger.info(f"🔍 model is bnb quantized: {_model_is_bnb_quantized}")
    logger.info(f"🔍 adapter is applied: {_adapter_is_applied}")

    # Tokenize SFT
    def preprocess_fn(examples):
        qs = examples["question"]
        ans = examples["answer"]
        feats = {"input_ids": [], "attention_mask": [], "labels": []}
        for q, a in zip(qs, ans):
            ex = _build_sft_features(tokenizer, q, a, args.max_length)
            feats["input_ids"].append(ex["input_ids"])
            feats["attention_mask"].append(ex["attention_mask"])
            feats["labels"].append(ex["labels"])
        return feats

    with accelerator.main_process_first():
        train_ds = train_raw.map(
            preprocess_fn,
            batched=True,
            remove_columns=train_raw.column_names,
            desc="Tokenizing GSM8K train (SFT)",
        )
        eval_ds = eval_raw.map(
            preprocess_fn,
            batched=True,
            remove_columns=eval_raw.column_names,
            desc="Tokenizing GSM8K eval (SFT)",
        )

    for idx in random.sample(range(len(train_ds)), k=min(2, len(train_ds))):
        logger.info(f"Sample {idx}: len={len(train_ds[idx]['input_ids'])}")

    # Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Load freeze dicts 
    freeze_rank_dict = {}
    freeze_alpha_dict = None
    if args.lora_adapter_dir is not None:
        fr_path = Path(args.lora_adapter_dir) / "freeze_rank_dict.json"
        if fr_path.exists():
            with open(fr_path, "r", encoding="utf-8") as f:
                freeze_rank_dict = json.load(f)

        fa_path = Path(args.lora_adapter_dir) / args.freeze_alpha_dict_filename
        if fa_path.exists():
            with open(fa_path, "r", encoding="utf-8") as f:
                freeze_alpha_dict = json.load(f)


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

            old_A = lora_A.weight.data.clone()
            old_B = lora_B.weight.data.clone()

            del module.lora_A
            del module.lora_B

            # frozen part
            module.lora_A_frozen = torch.nn.Linear(module.in_features, freeze_r, bias=False)
            module.lora_B_frozen = torch.nn.Linear(freeze_r, module.out_features, bias=False)
            module.lora_A_frozen.weight = torch.nn.Parameter(old_A[:freeze_r], requires_grad=True)
            module.lora_B_frozen.weight = torch.nn.Parameter(old_B[:, :freeze_r], requires_grad=True)

            # train part
            train_r = total_r - freeze_r
            module.lora_A_train = torch.nn.Linear(module.in_features, train_r, bias=False)
            module.lora_B_train = torch.nn.Linear(train_r, module.out_features, bias=False)
            module.lora_A_train.weight = torch.nn.Parameter(old_A[freeze_r:], requires_grad=True)
            module.lora_B_train.weight = torch.nn.Parameter(old_B[:, freeze_r:], requires_grad=True)

            def lora_forward_split(self, x):
                # base linear forward
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

        _, alpha_vec = alpha_cache[matched_key]

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

    # -----------------------------
    # Optimizer / scheduler
    # -----------------------------
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "ln_f.weight"]
    params_with_wd, params_without_wd = [], []
    names_with_wd, names_without_wd = [], []

    for pn, pp in model.named_parameters():
        if not pp.requires_grad:
            continue
        if any(nd in pn for nd in no_decay):
            params_without_wd.append(pp)
            names_without_wd.append(pn)
        else:
            params_with_wd.append(pp)
            names_with_wd.append(pn)

    if IS_MAIN:
        logger.info(f"Parameters with weight decay: {names_with_wd}")
        logger.info(f"Parameters without weight decay: {names_without_wd}")

    optimizer = torch.optim.AdamW(
        [{"params": params_with_wd, "weight_decay": args.weight_decay},
         {"params": params_without_wd, "weight_decay": 0.0}],
        lr=args.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    # Recompute after prepare
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
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
        if args.wandb-tags is not None:
            tracker_init_kwargs["wandb"]["tags"] = args.wandb-tags
        accelerator.init_trackers("gsm8k_train", experiment_config, tracker_init_kwargs)

    # -----------------------------
    # Training
    # -----------------------------
    logger.info("***** Running training *****")
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"  Num train examples = {len(train_ds)}")
    logger.info(f"  Num eval examples = {len(eval_ds)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (parallel & accum) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
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
            starting_epoch = resume_step // len(train_loader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_loader)

    progress_bar.update(completed_steps)

    def evaluate_loss():
        model.eval()
        losses = []
        for batch in eval_loader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.detach()))
        losses = torch.cat(losses)
        loss_mean = losses.mean().item()
        ppl = math.exp(min(20.0, loss_mean)) 
        return loss_mean, ppl

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_loader = accelerator.skip_first_batches(train_loader, resume_step)
        else:
            active_loader = train_loader

        for step, batch in enumerate(active_loader):
            outputs = model(**batch)
            loss = outputs.loss

            epoch_loss_sum += float(loss.detach().float().item())
            epoch_loss_count += 1

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            do_step = (step % args.gradient_accumulation_steps == 0) or (step == len(train_loader) - 1)
            if do_step:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking:
                    accelerator.log({"train_step_loss": loss.detach().float()}, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # Eval (loss/perplexity)
        eval_loss, eval_ppl = evaluate_loss()
        train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        if IS_MAIN:
            logger.info(f"epoch {epoch}: train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f} | eval_ppl={eval_ppl:.4f}")

        # Push intermediate checkpoints
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
        accelerator.end_training()

    # Save final
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

    if args.output_dir is not None and accelerator.is_main_process:
        # Save simple metrics
        eval_loss, eval_ppl = evaluate_loss()
        all_results = {"eval_loss": eval_loss, "eval_ppl": eval_ppl}
        with open(os.path.join(args.output_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
