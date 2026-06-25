"""
Generalized text-to-SQL eval: base or fine-tuned, any shot count.

Generation, SQL extraction, and scoring are identical across base 3-shot,
base 0-shot, and fine-tuned 0 shot, so accuracy deltas reflect the model,
not the harness.
"""

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


def load_model(model_id: str, adapter: str | None = None):
    """Load the model in 4-bit (NF4) with bf16 compute; optionally stack a LoRA adapter."""
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
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded adapter: {adapter}")
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
    ap = argparse.ArgumentParser(description="Text-to-SQL eval (base or fine-tuned)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None,
                    help="path to a LoRA adapter; omit for base-model eval")
    ap.add_argument("--val", default="data/spider/validation.json")
    ap.add_argument("--train", default="data/spider/train.json")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="run on first N val examples (smoke run)")
    ap.add_argument("--tag", default=None,
                    help="name for output files (default: <base|ft>_<shots>shot)")
    args = ap.parse_args()

    tag = args.tag or f"{'ft' if args.adapter else 'base'}_{args.shots}shot"

    val = prompt_utils.load_split(args.val)
    if args.limit:
        val = val[: args.limit]
    
    if args.shots > 0:
        train = prompt_utils.load_split(args.train)
        few = prompt_utils.select_few_shots(run_query, train, n=args.shots)
        print(f"using {len(few)} few-shot examples (cached in fewshots.json)")
    else:
        few = []
        print("zero shot: no few-shot examples")

    print(f"loading {args.model} ...")
    model, tok = load_model(args.model, args.adapter)

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

    os.makedirs("predictions", exist_ok=True)
    pred_path = f"predictions/{tag}.jsonl"
    with open(pred_path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"wrote predictions -> {pred_path}")

    report, results = evaluate(preds, run_query)
    report["model"] = args.model
    report["adapter"] = args.adapter
    report["shots"] = args.shots
    report["tag"] = tag

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{tag}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote report -> {report_path}")

    # Per-example scored results, so error analysis is a file to slice later.
    results_path = f"reports/{tag}_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"wrote per-example results -> {results_path}")

    print(f"\n==== {tag} ====")
    for k, v in report.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

