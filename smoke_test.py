import json
import sqlite3
from pathlib import Path

DATA = Path("data/spider")

def load_examples(split="train"):
    fname = "train.json" if split == "train" else "validation.json"
    return json.load(open(DATA / fname))

def db_path(db_id):
    return DATA / "database" / db_id / f"{db_id}.sqlite"

def run_query(db_id, sql):
    """Run a SQL string againsts its database, returns rows (or an error)."""
    con = sqlite3.connect(db_path(db_id))
    try:
        rows = con.execute(sql).fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        con.close()

if __name__ == "__main__":
    examples = load_examples("train")
    print(f"Loaded {len(examples)} training examples\n")

    # Run the gold SQL for first 5 examples
    for ex in examples[:5]:
        rows, err = run_query(ex["db_id"], ex["query"])
        print(f"DB:       {ex['db_id']}")
        print(f"Question: {ex['question']}")
        print(f"Gold SQL: {ex['query']}")
        if err:
            print(f"ERROR:      {err}")
        else:
            print(f"Result:     {rows[:3]}{' ...' if len(rows) > 3 else ''}")
        print("-" * 60)

