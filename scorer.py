"""
Execution-accuracy scorer for text-to-SQL.

Design Choices:
    
    1. Order sensitivity from GOLD query. If gold has ORDER BY, row
       order must match; otherwise rows are compared as multi-set (order-
       free, but duplicate sensitive).
    2. Column order is ignored. Accept any permutation of predicted columns
       that reproduces the gold result. This mirrors the offical Spider test-suite
       eval.
    3. Cell normalization. Numbers are coerced to float and rounded so that int-vs-float
       and float noise don't cause false mismatches. Bytes are decoded to str.
    4. A prediction that errors on execution is wrong, counted separately so we can see
       how much of the gap is malformed SQL vs. wrong-but-valid SQL.

    SCOPE: this is single-database execution accuracy. The official test-suite metric
    runs each query against several synthetic DB instances per schema to ctach queries
    that match the gold result only by coincidence on one database. That requires the
    test-suite databases (taoyds/test-suite-sql-eval) and could be bolted on later.
"""

from __future__ import annotations

import re
import json
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence

FLOAT_NDIGITS = 6 # rounding applied to numeric cells before comparsion
MAX_PERM_COLS = 6 # above this many columns, skip permutation search (positional compare)

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def gold_requires_order(gold_sql: str) -> bool:
    """Order matters iff the gold query contains ORDER BY."""

    return bool(_ORDER_BY_RE.search(gold_sql or ""))


def _norm_cell(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), FLOAT_NDIGITS)
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "replace")
        except Exception:
            return str(v)
    return v


def _norm_rows(rows: Sequence[Sequence[Any]]) -> list:
    return [tuple(_norm_cell(c) for c in row) for row in rows]


def _column_candidates(gold: list, pred: list, ncols: int):
    """For each gold column, the set of pred columns with a matching value-multiset"""

    gold_cols = [Counter(r[i] for r in gold) for i in range(ncols)]
    pred_cols = [Counter(r[i] for r in pred) for i in range(ncols)]
    cands = []

    for gi in range(ncols):
        c = [pi for pi in range(ncols) if gold_cols[gi] == pred_cols[pi]]
        if not c:
            return None
        cands.append(c)
    return cands


def _iter_column_perms(cands, ncols):
    """Backtracking over column assignments, pruned by candidates sets."""

    used = [False] * ncols
    assign = [0] * ncols

    def bt(gi):
        if gi == ncols:
            yield tuple(assign)
            return
        for pi in cands[gi]:
            if not used[pi]:
                used[pi] = True
                assign[gi] = pi
                yield from bt(gi + 1)
                used[pi] = False
    
    yield from bt(0)


def compare_results(gold_rows, pred_rows, order_matters: bool) -> bool:
    """True iff pred result set matches gold set under rules above"""

    gold = _norm_rows(gold_rows or [])
    pred = _norm_rows(pred_rows or [])

    if len(gold) != len(pred):
        return False
    if len(gold) == 0:
        return True
    ncols = len(gold[0])
    if len(pred[0]) != ncols:
        return False

    if ncols == 1:
        return gold == pred if order_matters else Counter(gold) == Counter(pred)

    if ncols > MAX_PERM_COLS:
        return gold == pred if order_matters else Counter(gold) == Counter(pred)

    cands = _column_candidates(gold, pred, ncols)
    if cands is None:
        return False

    gold_counter = None if order_matters else Counter(gold)
    for perm in _iter_column_perms(cands, ncols):
        pred_perm = [tuple(row[perm[i]] for i in range(ncols)) for row in pred]
        if order_matters:
            if gold == pred_perm:
                return True
        else:
            if gold_counter == Counter(pred_perm):
                return True

    return False


@dataclass
class ItemResult:
    db_id: str
    question: str
    gold: str
    pred: str
    correct: bool
    status: str # correct | pred_error | gold error | count_mismatch | content mismatch
    pred_error: Optional[str] = None
    gold_error: Optional[str] = None


def score_one(run_query, db_id, gold_sql, pred_sql, question="") -> ItemResult:
    gold_rows, gold_err = run_query(db_id, gold_sql)
    if gold_err is not None:
        return ItemResult(db_id, question, gold_sql, pred_sql, False,
                          "gold_error", gold_error=str(gold_err))
    
    pred_rows, pred_err = run_query(db_id, pred_sql)
    if pred_err is not None:
        return ItemResult(db_id, question, gold_sql, pred_sql, False,
                          "pred_err", pred_error=str(pred_err))
    
    order_matters = gold_requires_order(gold_sql)
    ok = compare_results(gold_rows, pred_rows, order_matters)
    if ok:
        status = "correct"
    elif len(gold_rows or []) != len(pred_rows or []):
        status = "count_mismatch"
    else:
        status = "content_mismatch"
    return ItemResult(db_id, question, gold_sql, pred_sql, ok, status)

def evaluate(predictions, run_query, verbose=True):
    """predictions: iterable of dicts with keys db_id, gold (or query), pred, question"""

    results = []
    for p in predictions:
        gold = p.get("gold", p.get("query"))
        results.append(score_one(run_query, p["db_id"], gold,
                             p.get("pred", ""), p.get("question", "")))
    n = len(results)
    correct = sum(r.correct for r in results)
    breakdown = Counter(r.status for r in results)
    acc = correct / n if n else 0.0
    report = {"n": n, "correct": correct,
              "execution_accuracy": acc, "breakdown": dict(breakdown)}
    if verbose:
        print(f"\nExecution accuracy: {acc:.4f} ({correct}/{n})")
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"    {k:16s} {v}")
    return report, results


def _load_run_query(module="smoke_test", func="run_query"):
    import importlib, sys, os
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module), func)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a text-to-SQL predictions .jsonl")
    ap.add_argument("predictions", help="path to predictions .jsonl "
                                         "(records: db_id, gold/query, pred)")
    ap.add_argument("--run-query-module", default="smoke_test")
    ap.add_argument("--out", default=None,
                    help="optional path to write per-item results .jsonl")
    args = ap.parse_args()

    run_query = _load_run_query(args.run_query_module)
    preds = [json.loads(l) for l in open(args.predictions) if l.strip()]
    report, results = evaluate(preds, run_query)

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"wrote per item results -> {args.out}")





