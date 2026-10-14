# B0 - the data contract between Track A (extraction) and Track B (modeling).
# Everything in Track B imports its column names from here, so a schema change is one edit.

ID_COLS = ["case_id", "pmid"]          # pmid = grouping key for cross-validation

NUMERIC = ["patient_age", "symptom_duration_months"]
CATEGORICAL = ["sex", "laterality", "pain_character",
               "MRI_finding", "offending_vessel", "nerve_morphological_change", "underlying_cause"]
BRANCH_FLAGS = ["branch_V1", "branch_V2", "branch_V3"]
TRIGGER_FLAGS = ["trig_chewing", "trig_touch", "trig_speaking", "trig_hygiene", "trig_cold_wind", "trig_other"]
TX_FLAGS = ["tx_drug_sodium_channel_blocker", "tx_drug_other", "tx_MVD", "tx_radiosurgery",
            "tx_percutaneous", "tx_other"]
# "not reported" is informative (case reports omit things non-randomly), so we keep explicit indicators
REPORTED_FLAGS = ["triggers_reported", "treatment_reported"]

TARGET_CLASS = "TN_classification"      # classic / idiopathic / secondary
TARGET_OUTCOME = "treatment_outcome"    # complete_relief / partial_relief / no_relief

CLASSES = ["classic", "idiopathic", "secondary"]
OUTCOMES = ["complete_relief", "partial_relief", "no_relief"]

# Imaging/etiology columns that DEFINE the Cruccu classes (classic = MRI compression with
# morphological change; secondary = underlying disease such as MS/tumour; idiopathic = neither).
# Using them to predict TN_classification is near-circular -> report with and without.
LEAKY_FOR_CLASSIFICATION = ["MRI_finding", "offending_vessel", "nerve_morphological_change", "underlying_cause"]

CLINICAL_FEATURES = (["patient_age", "symptom_duration_months", "sex", "laterality", "pain_character"]
                     + BRANCH_FLAGS + TRIGGER_FLAGS + ["triggers_reported"])

FEATURE_SETS = {
    # T1: predict TN subtype
    "T1_clinical_only": CLINICAL_FEATURES,
    "T1_with_imaging": CLINICAL_FEATURES + LEAKY_FOR_CLASSIFICATION,   # leakage ablation, expect inflated
    # T2: predict treatment outcome (subtype + imaging + treatment are legitimate inputs)
    "T2_outcome": CLINICAL_FEATURES + LEAKY_FOR_CLASSIFICATION + [TARGET_CLASS] + TX_FLAGS + ["treatment_reported"],
}

TASKS = {
    "T1": dict(target=TARGET_CLASS, labels=CLASSES, sets=["T1_clinical_only", "T1_with_imaging"]),
    "T2": dict(target=TARGET_OUTCOME, labels=OUTCOMES, sets=["T2_outcome"]),
}

CLEAN_COLUMNS = (ID_COLS + NUMERIC + ["sex", "laterality", "pain_character"] + BRANCH_FLAGS + TRIGGER_FLAGS
                 + REPORTED_FLAGS + ["MRI_finding", "offending_vessel", "nerve_morphological_change",
                                     "underlying_cause"] + TX_FLAGS + [TARGET_CLASS, TARGET_OUTCOME])


def categorical_in(features: list[str]) -> list[str]:
    """Categorical columns inside a feature list (TN_classification counts when used as an input)."""
    cats = set(CATEGORICAL) | {TARGET_CLASS}
    return [f for f in features if f in cats]


def numeric_in(features: list[str]) -> list[str]:
    return [f for f in features if f in NUMERIC]


def binary_in(features: list[str]) -> list[str]:
    return [f for f in features if f not in set(NUMERIC) | set(CATEGORICAL) | {TARGET_CLASS}]
