"""의료 도메인 적응 사전학습 (DAPT) - klue/roberta-large MLM 파인튜닝.

실행:
    python scripts/run_dapt.py
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("dapt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="klue/roberta-large")
    p.add_argument("--train-file", type=Path,
                   default=PROJECT_ROOT / "data" / "dapt_raw_train.txt")
    p.add_argument("--eval-file", type=Path,
                   default=PROJECT_ROOT / "data" / "dapt_raw_eval.txt")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "models" / "medical-roberta")
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--per-device-train-batch-size", type=int, default=16)
    p.add_argument("--per-device-eval-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preprocessing-num-workers", type=int, default=4)
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    for pth in (args.train_file, args.eval_file):
        if not pth.exists():
            logger.error("학습 파일이 없음: %s", pth)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("CUDA 사용 가능: %s (gpu=%d)", torch.cuda.is_available(),
                torch.cuda.device_count())
    logger.info("베이스 모델: %s", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("데이터셋 로드 중...")
    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.eval_file),
        },
    )
    logger.info(
        "train=%d, validation=%d",
        len(raw_datasets["train"]),
        len(raw_datasets["validation"]),
    )

    max_len = min(args.max_seq_length, tokenizer.model_max_length)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_special_tokens_mask=True,
        )

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info(
        "학습 시작 | effective_batch=%d | max_seq=%d | fp16=%s",
        args.per_device_train_batch_size * args.gradient_accumulation_steps,
        max_len,
        training_args.fp16,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("최종 evaluation 수행")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized["validation"])
    try:
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["perplexity"] = float("inf")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("모델 저장 완료: %s", args.output_dir)
    logger.info("최종 eval_loss=%.4f perplexity=%.4f",
                eval_metrics["eval_loss"], eval_metrics["perplexity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
