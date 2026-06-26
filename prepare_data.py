""" Build QLoRA training set as a prompt-completion JSONL """

from __future__ import annotations

import os
import json
import argparse

from tqdm import tqdm

from smoke_test import run_query
import prompt_utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/spider/train.json")
    ap.add_argument("--out", default="data/train_prompt_completion.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples (smoke run)")
    args = ap.parse_args()

    train = prompt_utils.load_split(args.train)
    if args.limit:
        train = train[: args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n, skipped = 0, 0
    with open(args.out, "w") as f:
        for ex in tqdm(train, desc="formatting"):
            try:
                prompt, completion = prompt_utils.build_prompt_completion(
                    run_query, ex["db_id"], ex["question"], ex["query"]
                )
            except RuntimeError:
                skipped += 1
                continue
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
            n += 1

    msg = f"wrote {n} examples -> {args.out}"
    if skipped:
        msg += f" ({skipped} skipped on schema-fetch errors)"
    print(msg)


if __name__ == "__main__":
    main()

