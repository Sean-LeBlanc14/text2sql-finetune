"""
QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct for text-to-SQL

Consumes data/train_prompt_completion.jsonl. Loss is computed on
the completion (the SQL) only. TRL's default for prompt-completion
datasets, so the schema+question is conditioning, not a target
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="data/train_prompt_completion.jsonl")
    p.add_argument("--output-dir", default="adapters/qwen2.5-coder-1.5b-spider")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="cap training examples (smoke runs)")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="cap optimizer steps (smoke runs, e.g. 20). -1 = use epochs")
    p.add_argument("--run-name", default="qlora-coder1.5b-spider")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", "text2sql-finetune")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA on all linear layers (QLoRA-paper)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2* args.lora_rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    ds = load_dataset("json", data_files=args.train_file, split="train")
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    # truncation_mode is keep_start -> a too-long example loses its SQL off
    # the END. Filter over-length rows instead of training on a broken label.
    def fits(ex):
        ids = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"],
                                            tokenize=True)
        return len(ids) <= args.max_length

    before = len(ds)
    ds = ds.filter(fits)
    dropped = before - len(ds)
    print(f"length filter: kept {len(ds)}/{before} "
          f"({dropped} dropped over {args.max_length} tokens)")
    if dropped > before * 0.05:
        print("  >5% dropped -- consider --max-length 1536")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=36,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="wandb",
        run_name=args.run_name,
        seed=42,
        loss_type="chunked_nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()

