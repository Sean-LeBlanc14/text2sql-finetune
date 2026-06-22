""" Generate few-shot predictions from a base model, then score them. """

from __future__ import annotations

import os
import re
import json
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from smoke_test import run_query
import prompt_utils
from scorer import evaluate

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def load_model(model_id: str):
    """Load the model in 4-bit (NF4) with bf16 compute"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tok

def extract_sql(text: str) -> str:
    """Pull clean, executable SQL out of the model's raw output"""

    t = text.strip()

    if "```" in t:
        m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(1).strip()

    t = re.sub(r"^\s*sql\s*:\s*", "", t, flags=re.IGNORECASE)

    if ";" in t:
        t = t.split(";")[0].strip() + ";"

    return " ".join(t.split())

@torch.inference_mode()
def generate_sql(model, tok, messages, max_new_tokens: int = 256):
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, #greedy: reproducible baseline
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = tok.decode(new_tokens, skip_special_tokens=True)
    return extract_sql(raw), raw

def main():
    ap = argparse.ArgumentParser(description="Baseline few-shot text-to-SQL eval")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--val", default="data/spider/validation.json")
    ap.add_argument("--train", default="data/spider/train.json")
    ap.add_argument("--n-shot", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="run on first N val examples (smoke run)")
    ap.add_argument("--out", default=None,
                    help="predictions path (default: predictions/baseline_<model>.jsonl)")
    args = ap.parse_args()

    val = prompt_utils.load_split(args.val)
    if args.limit:
        val = val[: args.limit]
    train = prompt_utils.load_split(args.train)

    few = prompt_utils.select_few_shots(run_query, train, n=args.n_shot)
    print(f"using {len(few)} few-shot examples (cached in fewshots.json)")

    print(f"loading {args.model} ...")
    model, tok = load_model(args.model)

    preds = []
    t0 = time.time()
    for ex in tqdm(val, desc="generating"):
        messages = prompt_utils.build_messages(
            run_query, ex["db_id"], ex["question"], few
        )
        sql, raw = generate_sql(model, tok, messages, args.max_new_tokens)
        preds.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold": ex["query"],
            "pred": sql,
            "raw": raw,
        })
    dt = time.time() - t0
    print(f"generated {len(preds)} predictions in {dt / 60:.1f} min")

    model_tag = args.model.split("/")[-1]
    out = args.out or f"predictions/baseline_{model_tag}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"wrote predictions -> {out}")

    report, results = evaluate(preds, run_query)

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/baseline_{model_tag}.json"
    report["model"] = args.model
    report["n_shot"] = args.n_shot
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote report -> {report_path}")


if __name__ == "__main__":
    main()

