# Track B - modeling & evaluation, built on mock data

By Abdelrahman El-Sherif.

Everything here runs **today** on synthetic data, so nothing waits for Track A.
When real data lands: `python b1_normalize.py extracted_structured.csv` then `python b4_run_mock.py clean_dataset.csv`.

```
pip install -r ../requirements.txt
python -m pytest -q             # ~5 s, 8 tests
python b4_run_mock.py           # ~90 s, writes outputs/ (tables + PNGs)
```

| file | job |
|---|---|
| `b0_schema.py` | the data contract: column names, feature sets, which columns are leaky. One edit point. |
| `b1_normalize.py` | Track A `extracted_structured.csv` (v1 free text or v2 controlled vocab) -> clean table. Prints an `unmapped_report`; extend the keyword lists when it is not "none". |
| `b1_make_mock_data.py` | SYNTHETIC data in the clean schema. All probabilities are invented placeholders. |
| `b2_pipeline.py` | preprocessing (median+missing-indicator for numbers, one-hot with "not_reported" as its own level), models, **grouped** CV splits. |
| `b3_evaluate.py` | repeated stratified group CV, macro-F1 / balanced acc / accuracy, pooled confusion matrices, per-class table, permutation importance, dummy baseline check, shuffled-label sanity check. |
| `b4_run_mock.py` | the whole thing end to end. |
| `b6_plots.py` | confusion-matrix and importance PNGs. |
| `test_track_b.py` | tests for parsers, normalizer, leak checks, pipeline behaviour. |

## Design decisions (and why)

1. **Split by paper, not by patient.** A case series gives several rows per PMID; random splits would put siblings in train and test. `StratifiedGroupKFold` on `pmid`, repeated 5x with different seeds; `grouped_splits` raises if a paper ever straddles.
2. **Two tasks, three feature sets.**
   - `T1_clinical_only`: predict TN subtype from age, sex, side, branch, pain type, triggers. This is the defensible research question.
   - `T1_with_imaging`: adds MRI finding / vessel / nerve change / underlying cause. These *define* the Cruccu subtypes, so this is close to circular. Kept only as an ablation to show the inflation (mock: ~0.96 macro-F1 vs ~0.5).
   - `T2_outcome`: predict relief (complete / partial / none) from everything known before treatment plus the treatment given.
3. **"Not reported" is signal, not noise.** Categorical gaps stay a level; numeric gaps get median + indicator; `triggers_reported` / `treatment_reported` flags separate "none" from "not stated".
4. **Always report against a dummy.** `dummy_stratified` and `dummy_most_frequent` are in every table; accuracy alone is misleading with imbalance (most-frequent gets ~0.53 accuracy on mock but 0.23 macro-F1).
5. **Sanity check = shuffled labels.** Every model must drop to dummy level. If one does not, there is a leak.
6. **Importance via permutation on raw columns**, not forest impurity (biased toward high-cardinality / numeric columns), and it reports per original column rather than per one-hot column.

## Not done yet (waiting on real data or a decision)

- B6 SHAP: add once n is known; permutation importance is the baseline.
- B7 class imbalance: `class_weight="balanced"` for now. If you add SMOTE, do it inside the CV pipeline (`imblearn.pipeline.Pipeline`), never before splitting.
- Hyperparameter tuning: with a few hundred rows use nested CV or leave defaults and say so.
- Confidence intervals: fold std is reported; add a paper-level bootstrap when the real n is known.
- Outcome definition for T2 is a placeholder (3 levels). Decide before the real run.
