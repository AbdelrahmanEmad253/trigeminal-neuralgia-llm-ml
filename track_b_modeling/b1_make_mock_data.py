# B1b - SYNTHETIC data shaped like the real clean table (b0_schema.CLEAN_COLUMNS).
#
# !! Every number below is an invented placeholder so the pipeline can be built and tested
# !! before real data exists. None of it is clinical knowledge. Results on this data mean
# !! nothing about trigeminal neuralgia - they only show the code runs and has no leaks.
#
# Built-in properties on purpose (so the pipeline's checks have something to find):
#   - class imbalance
#   - patients of one paper tend to share a subtype (so grouping CV by paper matters)
#   - subtype-defining imaging columns are strongly linked to the label (leakage ablation)
#   - "not_reported" gaps, more for some fields than others

import numpy as np
import pandas as pd
from b0_schema import CLEAN_COLUMNS, CLASSES

NR = "not_reported"


def _choice(rng, options, probs):
    probs = np.array(probs, dtype=float)
    return options[rng.choice(len(options), p=probs / probs.sum())]


def _gap(rng, value, p):
    return NR if rng.random() < p else value


def make_mock(n_papers=200, seed=0, prior=(0.50, 0.25, 0.25), within_paper_agreement=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_papers):
        pmid = str(10_000_000 + p)
        n_pat = 1 if rng.random() < 0.75 else int(rng.integers(2, 5))
        paper_cls = CLASSES[rng.choice(3, p=prior)]
        for k in range(n_pat):
            cls = paper_cls if rng.random() < within_paper_agreement else CLASSES[rng.choice(3, p=prior)]
            rows.append(_patient(rng, f"{pmid}_{k}", pmid, cls))
    df = pd.DataFrame(rows)[CLEAN_COLUMNS]
    return df


def _patient(rng, case_id, pmid, cls):
    sec, classic = cls == "secondary", cls == "classic"
    age = float(np.clip(rng.normal({"classic": 62, "idiopathic": 58, "secondary": 45}[cls], 12), 18, 90))
    dur = float(np.exp(rng.normal(3.0, 1.0)))                       # months
    laterality = _choice(rng, ["right", "left", "bilateral"], [0.5, 0.5, 0.3 if sec else 0.03])
    pain = _choice(rng, ["paroxysmal", "mixed", "continuous"],
                   {"classic": [7, 3, 0.2], "idiopathic": [6, 4, 0.3], "secondary": [4.5, 4.5, 1]}[cls])
    branch = {"branch_V1": int(rng.random() < (0.35 if sec else 0.12)),
              "branch_V2": int(rng.random() < 0.55), "branch_V3": int(rng.random() < 0.55)}
    if sum(branch.values()) == 0:
        branch["branch_V2"] = 1

    trig_reported = rng.random() > 0.2
    trig = {f"trig_{t}": int(trig_reported and rng.random() < pr) for t, pr in
            [("chewing", 0.4), ("touch", 0.4), ("speaking", 0.2), ("hygiene", 0.25), ("cold_wind", 0.1), ("other", 0.05)]}

    if classic:
        mri = _choice(rng, ["compression_yes", "compression_no", NR], [9, 0.5, 0.5])
        vessel = _choice(rng, ["superior_cerebellar_artery", "aica", "vein", "other", NR], [6.5, 1, 1.5, 1, 0.5])
        morph = _choice(rng, ["yes", "no", NR], [7, 1, 2])
        cause = _choice(rng, ["none_identified", NR], [7, 3])
    elif cls == "idiopathic":
        mri = _choice(rng, ["compression_no", "compression_yes", NR], [8, 1, 1])
        vessel = _choice(rng, ["none", "vein", NR], [6, 1, 3])
        morph = _choice(rng, ["no", "yes", NR], [6, 0.3, 3.7])
        cause = _choice(rng, ["none_identified", NR], [8, 2])
    else:
        mri = _choice(rng, ["other_lesion", "compression_no", NR], [8.5, 1, 0.5])
        vessel = _choice(rng, ["none", NR, "other"], [4, 5, 1])
        morph = _choice(rng, ["no", NR, "yes"], [3, 6, 1])
        cause = _choice(rng, ["multiple_sclerosis", "tumor", "other"], [5, 4, 1])

    # treatment + outcome (invented dependence: surgery > drug, classic > secondary)
    tx = {k: 0 for k in ["tx_drug_sodium_channel_blocker", "tx_drug_other", "tx_MVD", "tx_radiosurgery", "tx_percutaneous", "tx_other"]}
    tx["tx_drug_sodium_channel_blocker"] = int(rng.random() < 0.7)
    tx["tx_drug_other"] = int(rng.random() < 0.2)
    tx["tx_MVD"] = int(rng.random() < (0.5 if classic else 0.15))
    tx["tx_radiosurgery"] = int(rng.random() < 0.12)
    tx["tx_percutaneous"] = int(rng.random() < 0.1)
    tx["tx_other"] = int(rng.random() < 0.05)
    treat_reported = rng.random() > 0.1
    if not treat_reported:
        tx = {k: 0 for k in tx}
    score = (1.2 * tx["tx_MVD"] + 0.6 * tx["tx_radiosurgery"] + 0.4 * tx["tx_percutaneous"] + 0.3 * tx["tx_drug_sodium_channel_blocker"]
             + (0.4 if classic else -0.5 if sec else 0.0) + rng.normal(0, 0.8))
    outcome = "complete_relief" if score > 1.0 else "partial_relief" if score > 0.2 else "no_relief"
    outcome = _gap(rng, outcome, 0.25)

    return {
        "case_id": case_id, "pmid": pmid,
        "patient_age": _nan_gap(rng, round(age), 0.08), "symptom_duration_months": _nan_gap(rng, round(dur, 1), 0.35),
        "sex": _gap(rng, "female" if rng.random() < 0.58 else "male", 0.05),
        "laterality": _gap(rng, laterality, 0.15), "pain_character": _gap(rng, pain, 0.1),
        **branch, **trig, "triggers_reported": int(trig_reported), "treatment_reported": int(treat_reported),
        "MRI_finding": mri, "offending_vessel": vessel, "nerve_morphological_change": morph, "underlying_cause": cause,
        **tx,
        "TN_classification": _gap(rng, cls, 0.10), "treatment_outcome": outcome,
    }


def _nan_gap(rng, value, p):
    return np.nan if rng.random() < p else value


if __name__ == "__main__":
    df = make_mock()
    df.to_csv("mock_clean_dataset.csv", index=False)
    print(df.shape, "-> mock_clean_dataset.csv (SYNTHETIC)")
    print(df["TN_classification"].value_counts())
