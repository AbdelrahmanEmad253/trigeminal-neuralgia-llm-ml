# B4 - end-to-end run. By default on MOCK data (to prove the pipeline); pass a real clean CSV to run on real data.
#   python b4_run_mock.py                       -> mock data
#   python b4_run_mock.py clean_dataset.csv     -> real data (output of b1_normalize.py)
import os, sys
import pandas as pd
from b0_schema import TASKS
from b1_make_mock_data import make_mock
from b2_pipeline import get_models
from b3_evaluate import evaluate, check_against_dummy
from b6_plots import plot_confusion, plot_importance
from sklearn.ensemble import RandomForestClassifier

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
REAL = len(sys.argv) > 1
df = pd.read_csv(sys.argv[1], dtype={"pmid": str}) if REAL else make_mock(seed=1)
tag = "" if REAL else "MOCK DATA - "
prefix = "real" if REAL else "mock"
if not REAL:
    df.to_csv(f"{OUT}/mock_clean_dataset.csv", index=False)
    print("!! MOCK (synthetic) data: scores below say nothing about trigeminal neuralgia !!")

all_summaries = []
for task, spec in TASKS.items():
    for fs in spec["sets"]:
        print(f"\n===== {task} / {fs} =====")
        r = evaluate(df, task, fs, n_repeats=5)
        print(f"cases={r['n_cases']} papers={r['n_papers']} class_counts={r['class_counts']}")
        print(r["summary"].to_string())
        beats, dummy = check_against_dummy(r["summary"])
        print(f"best dummy macro-F1={dummy:.3f}; models beating it by >0.05: {beats}")
        s = r["summary"].reset_index(); s.insert(0, "feature_set", fs); s.insert(0, "task", task)
        all_summaries.append(s)
        r["per_class"].to_csv(f"{OUT}/{prefix}_{fs}_per_class.csv")
        best = r["summary"].drop(index=[i for i in r["summary"].index if i.startswith("dummy")]).index[0]
        plot_confusion(r["confusion"][best], f"{tag}{fs}: {best} (pooled out-of-fold)", f"{OUT}/{prefix}_{fs}_confusion.png")

        # permutation importance on one repeat with a lighter forest (B6)
        ri = evaluate(df, task, fs, n_repeats=1, importance_for=("random_forest",), perm_repeats=3,
                      models={"random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                                      class_weight="balanced_subsample", random_state=0)})
        imp = ri["importance"].loc["random_forest"]["mean"]
        imp.to_csv(f"{OUT}/{prefix}_{fs}_importance.csv")
        plot_importance(imp, f"{tag}{fs}: permutation importance (random forest)", f"{OUT}/{prefix}_{fs}_importance.png")

# sanity check: shuffled labels must fall to dummy level
print("\n===== SANITY: shuffled labels (T1_with_imaging) =====")
rs = evaluate(df, "T1", "T1_with_imaging", n_repeats=3, shuffle_labels=True)
print(rs["summary"].to_string())
beats, dummy = check_against_dummy(rs["summary"])
print("models beating dummy on shuffled labels (all must be False):", beats)
print("SANITY", "PASS" if not any(beats.values()) else "FAIL - investigate a leak")

pd.concat(all_summaries).to_csv(f"{OUT}/{prefix}_results_summary.csv", index=False)
print(f"\nsaved tables + plots to ./{OUT}/")
