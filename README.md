# Text-to-SQL Fine-Tune

## Hypothesis
QLoRA fine-tuning Qwen2.5-Coder-1.5B-Instruct improves execution accuracy on the
Spider dataset over a few-shot baseline of **57.25%** (single-database, val set).

## Metric
Execution accuracy: does the generated SQL run and return the correct rows?
Scored by executing predicted and gold SQL and comparing result sets, with
column-permutation matching, order/multiset handling, and numeric normalization
(`scorer.py`, unit-tested in `test_scorer.py`). Single-database execution
accuracy — the official test-suite metric runs multiple DB instances per schema
and is a possible later upgrade.

## Baseline
Qwen2.5-Coder-1.5B-Instruct, 3-shot, Spider dev (1,034 examples): **57.25%**.
Dominant failure: schema grounding — 244/257 execution errors are `no such
column` (inventing or misattributing column names), the gap fine-tuning targets.

## Status
- [x] baseline — 57.25%
- [ ] first fine-tune
- [ ] eval
- [ ] ablations 
