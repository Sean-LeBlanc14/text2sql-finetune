# Text-to-SQL Fine-Tune

## Hypothesis
QLoRA fine-tuning Qwen2.5-Coder-1.5B-Instruct improves execution accuracy on the
Spider dataset over a zero-shot baseline of **58.22%** (single-database, val set).
**Confirmed:** one epoch lifts execution accuracy to **80.08%** (+21.9 pts).

## Result
Qwen2.5-Coder-1.5B-Instruct, Spider val (1,034 examples), greedy decoding:

| config        | execution accuracy    |
|---------------|-----------------------|
| ft, 0-shot    | **80.08%** (828/1034) |
| base, 0-shot  | 58.22% (602/1034)     |
| base, 3-shot  | 57.25% (592/1034)     |

One epoch of QLoRA fine-tuning raised execution accuracy from 58.22% to 80.08%
(+21.9 pts) over the base model's best configuration. The gain is concentrated in
execution errors: `pred_err` fell **252 → 66** (−74%), confirming the fine-tune
closed the schema-grounding gap diagnosed at baseline. Mismatches also dropped
(count 104 → 90, content 76 → 50).

| failure bucket   | base 0-shot | ft 0-shot |
|------------------|-------------|-----------|
| pred_err         | 252         | 66        |
| count_mismatch   | 104         | 90        |
| content_mismatch | 76          | 50        |

## Metric
Execution accuracy: does the generated SQL run and return the correct rows?
Scored by executing predicted and gold SQL and comparing result sets, with
column-permutation matching, order/multiset handling, and numeric normalization
(`scorer.py`, unit-tested in `test_scorer.py`). Single-database execution
accuracy — the official test-suite metric runs multiple DB instances per schema
and is a possible later upgrade.

## Baseline
Few-shot exemplars did not help on this instruct/code model — 0-shot edges out
3-shot — so training and eval are both zero-shot, and 58.22% was the bar to beat.
Dominant baseline failure was schema grounding: ~250 of the misses were queries
that don't execute (`no such column` — inventing or misattributing column names),
the gap fine-tuning targeted. Train and eval share one prompt builder
(`prompt_utils.build_messages`) and one scorer, so base vs. fine-tuned differ only
in weights — the delta is not a prompt artifact.

## Status
- [x] baseline — base 0-shot 58.22%, base 3-shot 57.25%
- [x] training set built — 9,702 examples, 0 dropped
- [x] first fine-tune — QLoRA, rank 16, 1 epoch
- [x] eval vs baseline — ft 0-shot 80.08% (+21.9 pts)
- [ ] ablations — rank sweep, epoch sweep, data-size curve
- [ ] error analysis — slice remaining 206 failures by query type
