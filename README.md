# Text-to-SQL Fine-Tune

## Hypothesis
QLoRA fine-tuning Qwen2.5-Coder-1.5B-Instruct improves execution accuracy on the
Spider dataset over a zero-shot baseline of **58.22%** (single-database, val set).

## Metric
Execution accuracy: does the generated SQL run and return the correct rows?
Scored by executing predicted and gold SQL and comparing result sets, with
column-permutation matching, order/multiset handling, and numeric normalization
(`scorer.py`, unit-tested in `test_scorer.py`). Single-database execution
accuracy — the official test-suite metric runs multiple DB instances per schema
and is a possible later upgrade.

## Baseline
Qwen2.5-Coder-1.5B-Instruct, Spider val (1,034 examples), greedy decoding:

| config        | execution accuracy |
|---------------|--------------------|
| base, 0-shot  | **58.22%** (602/1034) |
| base, 3-shot  | 57.25% (592/1034)  |

Few-shot exemplars did not help on this instruct/code model — 0-shot edges out
3-shot — so training and eval are both zero-shot, and **58.22% is the bar to
beat**. Dominant failure is schema grounding: ~250 of the misses are queries that
don't execute (`no such column` — inventing or misattributing column names), the
gap fine-tuning targets. The remainder run but return wrong rows (count/content
mismatch).

## Status
- [x] baseline — base 0-shot 58.22%, base 3-shot 57.25%
- [ ] training set built
- [ ] first fine-tune
- [ ] eval vs baseline
- [ ] ablations
