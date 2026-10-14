# B2 - preprocessing, models, and leak-free cross-validation splits.

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from b0_schema import FEATURE_SETS, TASKS, categorical_in, numeric_in, binary_in

NOT_USABLE_LABELS = {"not_reported", ""}


def prepare_task(df, task, feature_set):
    """Rows with a usable label -> (X, y, groups, labels). Drops 'not_reported'/unmapped targets."""
    spec = TASKS[task]
    if feature_set not in spec["sets"]:
        raise ValueError(f"{feature_set} is not a feature set of task {task}: {spec['sets']}")
    d = df[df[spec["target"]].isin(spec["labels"])].reset_index(drop=True)
    features = FEATURE_SETS[feature_set]
    if spec["target"] in features:
        raise AssertionError("target is inside its own feature list (leak)")
    X = d[features].copy()
    for c in categorical_in(features):
        X[c] = X[c].fillna("not_reported").astype(str)       # missing category = its own, informative level
    return X, d[spec["target"]].astype(str), d["pmid"].astype(str), spec["labels"]


def build_preprocessor(features):
    num, cat, binary = numeric_in(features), categorical_in(features), binary_in(features)
    return ColumnTransformer([
        # NaN age/duration -> median + a 0/1 "was missing" column (missingness is informative here)
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("bin", "passthrough", binary),
    ], sparse_threshold=0.0)


def get_models(seed=0):
    """Estimators WITHOUT preprocessing; make_pipeline() attaches it."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", random_state=seed),
    }


def make_pipeline(model, features):
    return Pipeline([("prep", build_preprocessor(features)), ("model", clone(model))])


def grouped_splits(y, groups, n_splits=5, n_repeats=5, seed=0):
    """Repeated stratified GROUP k-fold: patients of one paper never straddle train/test.

    Yields (repeat, fold, train_idx, test_idx). n_splits is lowered automatically if the
    rarest class has too few papers.
    """
    papers_per_class = pd.DataFrame({"y": y.values, "g": groups.values}).groupby("y")["g"].nunique()
    n_splits = int(min(n_splits, papers_per_class.min()))
    if n_splits < 2:
        raise ValueError(f"rarest class has only {papers_per_class.min()} paper(s); cannot cross-validate")
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # sklearn warns when a class is rare in a fold
            for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
                if set(groups.iloc[tr]) & set(groups.iloc[te]):
                    raise AssertionError("paper appears in both train and test (group leak)")
                yield r, f, tr, te
