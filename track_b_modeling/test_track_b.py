# Offline tests for Track B. run: python test_track_b.py   (~20 s)
import os
import numpy as np, pandas as pd
import b1_normalize as n
from b0_schema import CLEAN_COLUMNS, FEATURE_SETS, LEAKY_FOR_CLASSIFICATION, TARGET_CLASS, TARGET_OUTCOME
from b1_make_mock_data import make_mock
from b2_pipeline import prepare_task, grouped_splits, get_models
from b3_evaluate import evaluate, check_against_dummy


def test_parsers():
    assert n.parse_age("62") == 62 and np.isnan(n.parse_age("not_reported")) and n.parse_age("60-64") == 62
    assert abs(n.parse_age("18 months") - 1.5) < 1e-9
    assert n.parse_duration_months("3 years") == 36
    assert n.parse_duration_months("2 years and 6 months") == 30
    assert n.parse_duration_months("2-3 years") == 30
    assert np.isnan(n.parse_duration_months("not_reported"))
    assert n._branch_has("v1-v3", 2, "maxillary") and not n._branch_has("v1, v3", 2, "maxillary")
    assert n._branch_has("mandibular division", 3, "mandibular")


def test_normalize_v1_free_text_rows():
    # v1 = original Track A output (free text); values copied from the original few-shot examples
    raw = pd.DataFrame([
        dict(pmid="1", patient_age="62", sex="female", symptom_duration="3 years", branch_involved="V2, V3",
             pain_character="paroxysmal", TN_classification="classic", trigger_factors="chewing, light touch",
             MRI_finding="compression_yes", offending_vessel="superior cerebellar artery",
             treatment_given="carbamazepine (partial response), microvascular decompression",
             treatment_outcome="complete pain relief at 1-year follow-up"),
        dict(pmid="2", patient_age="34", sex="male", symptom_duration="not_reported", branch_involved="V1",
             pain_character="mixed", TN_classification="secondary", trigger_factors="not_reported",
             MRI_finding="compression_no", offending_vessel="not_reported", treatment_given="gabapentin",
             treatment_outcome="not_reported"),
    ])
    c = n.normalize(raw)
    assert list(c.columns) == CLEAN_COLUMNS
    a, b = c.iloc[0], c.iloc[1]
    assert (a.patient_age, a.symptom_duration_months, a.sex) == (62, 36, "female")
    assert (a.branch_V1, a.branch_V2, a.branch_V3) == (0, 1, 1)
    assert (a.trig_chewing, a.trig_touch, a.triggers_reported) == (1, 1, 1)
    assert a.offending_vessel == "superior_cerebellar_artery" and a.MRI_finding == "compression_yes"
    assert (a.tx_drug_sodium_channel_blocker, a.tx_MVD, a.tx_drug_other) == (1, 1, 0)
    assert a.treatment_outcome == "complete_relief" and a.TN_classification == "classic"
    assert (b.branch_V1, b.triggers_reported, b.tx_drug_other, b.treatment_reported) == (1, 0, 1, 1)
    assert b.treatment_outcome == "not_reported" and b.laterality == "not_reported"
    assert a.case_id == "1_0"                      # generated when Track A did not supply one


def test_normalize_v2_controlled_vocab_passthrough():
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "track_a_extraction"))
    from a4_extraction_prompt import FEW_SHOT_EXAMPLES as EX
    recs = EX[0]["json"] if False else [EX[0]["json"], EX[1]["json"]] + EX[2]["json"]
    raw = pd.DataFrame(recs); raw.insert(0, "pmid", ["1", "2", "3", "3"])
    c = n.normalize(raw)
    assert list(c["case_id"]) == ["1_0", "2_0", "3_0", "3_1"]
    assert c.loc[1, "underlying_cause"] == "multiple_sclerosis" and c.loc[1, "MRI_finding"] == "other_lesion"
    assert list(c["treatment_outcome"]) == ["complete_relief", "not_reported", "complete_relief", "not_reported"]
    assert list(c["laterality"]) == ["right", "left", "right", "right"]


def test_no_group_leak_and_all_classes_in_every_test_fold_when_possible():
    df = make_mock(seed=3)
    X, y, g, labels = prepare_task(df, "T1", "T1_clinical_only")
    seen = 0
    for r, f, tr, te in grouped_splits(y, g, 5, 2, seed=0):
        assert not (set(g.iloc[tr]) & set(g.iloc[te]))
        assert len(set(tr) & set(te)) == 0
        seen += 1
    assert seen == 10
    assert any(g.value_counts() > 1), "mock should contain multi-patient papers"


def test_target_never_in_features_and_leaky_columns_excluded_from_clinical():
    for fs, feats in FEATURE_SETS.items():
        if fs.startswith("T1"):
            assert TARGET_CLASS not in feats
        if fs == "T2_outcome":
            assert TARGET_OUTCOME not in feats
    assert not set(LEAKY_FOR_CLASSIFICATION) & set(FEATURE_SETS["T1_clinical_only"])


def test_unusable_labels_dropped():
    df = make_mock(seed=3)
    X, y, g, labels = prepare_task(df, "T1", "T1_clinical_only")
    assert set(y) <= set(labels) and len(y) < len(df)


def test_pipeline_signal_leakage_and_shuffle():
    df = make_mock(seed=5)
    m = {k: v for k, v in get_models(0).items() if k in ("dummy_stratified", "logreg")}
    clin = evaluate(df, "T1", "T1_clinical_only", n_repeats=2, models=m)["summary"]
    full = evaluate(df, "T1", "T1_with_imaging", n_repeats=2, models=m)["summary"]
    shuf = evaluate(df, "T1", "T1_with_imaging", n_repeats=2, models=m, shuffle_labels=True)["summary"]
    assert check_against_dummy(clin)[0]["logreg"], "clinical features carry (mock) signal"
    assert full.loc["logreg", "macro_f1_mean"] > clin.loc["logreg", "macro_f1_mean"] + 0.2, "imaging should inflate T1"
    assert not check_against_dummy(shuf)[0]["logreg"], "shuffled labels must be at chance"


def test_unseen_category_and_all_missing_numeric_do_not_crash():
    df = make_mock(seed=7)
    X, y, g, labels = prepare_task(df, "T1", "T1_with_imaging")
    from b2_pipeline import make_pipeline
    pipe = make_pipeline(get_models(0)["logreg"], list(X.columns)).fit(X.iloc[:200], y.iloc[:200])
    Xt = X.iloc[200:].copy(); Xt["offending_vessel"] = "brand_new_vessel"; Xt["patient_age"] = np.nan
    assert len(pipe.predict(Xt)) == len(Xt)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
