# B3 - cross-validated evaluation, confusion matrices, sanity checks, feature importance.

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from b2_pipeline import prepare_task, get_models, make_pipeline, grouped_splits
from b0_schema import FEATURE_SETS


def evaluate(df, task, feature_set, n_splits=5, n_repeats=5, seed=0,
             models=None, shuffle_labels=False, importance_for=(), perm_repeats=3):
    """Repeated stratified group CV.

    Returns dict(fold_scores, summary, confusion, per_class, importance, n_cases, n_papers, class_counts).
    shuffle_labels=True is the sanity check: with labels shuffled, every model must fall to dummy level.
    """
    X, y, groups, labels = prepare_task(df, task, feature_set)
    if shuffle_labels:
        y = pd.Series(np.random.default_rng(seed).permutation(y.values), index=y.index)
    models = models or get_models(seed)
    feats = FEATURE_SETS[feature_set]

    fold_rows, imp_rows = [], []
    conf = {m: np.zeros((len(labels), len(labels)), dtype=int) for m in models}
    pc_rows = []

    for r, f, tr, te in grouped_splits(y, groups, n_splits, n_repeats, seed):
        for name, est in models.items():
            pipe = make_pipeline(est, feats).fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            yt = y.iloc[te]
            fold_rows.append(dict(
                model=name, repeat=r, fold=f, n_test=len(te),
                accuracy=accuracy_score(yt, pred),
                balanced_accuracy=balanced_accuracy_score(yt, pred),
                macro_f1=f1_score(yt, pred, labels=labels, average="macro", zero_division=0)))
            conf[name] += confusion_matrix(yt, pred, labels=labels)
            p, rc, f1, sup = precision_recall_fscore_support(yt, pred, labels=labels, zero_division=0)
            for i, lab in enumerate(labels):
                pc_rows.append(dict(model=name, repeat=r, fold=f, label=lab, precision=p[i], recall=rc[i], f1=f1[i], support=sup[i]))
            if name in importance_for and not shuffle_labels and len(yt.unique()) > 1:
                pi = permutation_importance(pipe, X.iloc[te], yt, scoring="f1_macro", n_repeats=perm_repeats, random_state=seed)
                for col, v in zip(X.columns, pi.importances_mean):
                    imp_rows.append(dict(model=name, repeat=r, fold=f, feature=col, importance=v))

    fold_scores = pd.DataFrame(fold_rows)
    summary = (fold_scores.groupby("model")[["macro_f1", "balanced_accuracy", "accuracy"]]
               .agg(["mean", "std"]).round(3))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.sort_values("macro_f1_mean", ascending=False)

    per_class = (pd.DataFrame(pc_rows).groupby(["model", "label"])[["precision", "recall", "f1"]].mean().round(3))
    importance = None
    if imp_rows:
        importance = (pd.DataFrame(imp_rows).groupby(["model", "feature"])["importance"]
                      .agg(["mean", "std"]).sort_values("mean", ascending=False).round(4))

    return dict(fold_scores=fold_scores, summary=summary,
                confusion={m: pd.DataFrame(c, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
                           for m, c in conf.items()},
                per_class=per_class, importance=importance,
                n_cases=len(y), n_papers=groups.nunique(), class_counts=y.value_counts().to_dict(), labels=labels)


def check_against_dummy(summary: pd.DataFrame, margin: float = 0.05) -> tuple[dict[str, bool], float]:
    """Models that don't beat the best dummy macro-F1 by `margin` carry no usable signal."""
    dummy_best = summary.loc[summary.index.str.startswith("dummy"), "macro_f1_mean"].max()
    real = summary.loc[~summary.index.str.startswith("dummy")]
    return {m: bool(v > dummy_best + margin) for m, v in real["macro_f1_mean"].items()}, float(dummy_best)
