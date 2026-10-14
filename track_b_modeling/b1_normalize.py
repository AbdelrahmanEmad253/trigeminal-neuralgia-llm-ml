# B1a - turn Track A's extracted_structured.csv (v1 free text OR v2 controlled vocab)
# into the clean modeling table defined in b0_schema.CLEAN_COLUMNS.
#
# These are keyword HEURISTICS for free-text fields. Always look at the `unmapped_report`
# it prints, extend the keyword lists, and spot-check ~20 rows by hand.

import re
import numpy as np
import pandas as pd
from b0_schema import CLEAN_COLUMNS, TARGET_CLASS, TARGET_OUTCOME

NR = "not_reported"


def _s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip().lower()


def parse_age(x):
    s = _s(x)
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return np.nan
    age = float(np.mean(nums[:2]))
    if "month" in s and "year" not in s:
        age /= 12
    return age if 0 <= age <= 110 else np.nan


def parse_duration_months(x):
    s = _s(x)
    if not s or s == NR:
        return np.nan
    mult = {"year": 12, "yr": 12, "month": 1, "mo": 1, "week": 12 / 52, "wk": 12 / 52, "day": 12 / 365}
    # range like "2-3 years" -> midpoint
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(year|yr|month|mo|week|wk|day)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * mult[m.group(3)]
    # "3 years", "2 years and 6 months" -> sum the parts
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(year|yr|month|mo|week|wk|day)", s)
    if parts:
        return sum(float(n) * mult[u] for n, u in parts)
    return np.nan


def _branch_has(s, n, name):
    """True if division n (1-3) is mentioned: 'V2', 'V2, V3', 'V1-V3' (range), or 'maxillary'."""
    if name in s:
        return True
    for a, b in re.findall(r"v\s*([123])\s*(?:-|–|to)\s*v?\s*([123])", s):
        if int(a) <= n <= int(b):
            return True
    return bool(re.search(rf"\bv\s*{n}\b", s))


def pick(x, mapping, default=NR):
    s = _s(x).replace(" ", "_")
    return mapping.get(s, default if s in ("", NR) else "unmapped:" + s)


def multi_hot(text, keyword_map):
    s = _s(text)
    return {col: int(any(k in s for k in kws)) for col, kws in keyword_map.items()}


TRIGGER_KW = {
    "trig_chewing": ["chew", "eating", "mastic"], "trig_touch": ["touch", "tactile", "light contact"],
    "trig_speaking": ["speak", "talk"], "trig_hygiene": ["brush", "wash", "shav", "teeth", "face"],
    "trig_cold_wind": ["cold", "wind", "breeze"],
}
TX_KW = {
    "tx_drug_sodium_channel_blocker": ["carbamazepine", "oxcarbazepine", "phenytoin", "lamotrigine"],
    "tx_drug_other": ["gabapentin", "pregabalin", "baclofen", "medication", "drug", "analgesic", "amitriptyline"],
    "tx_MVD": ["microvascular decompression", "mvd", "decompression"],
    "tx_radiosurgery": ["gamma knife", "radiosurgery", "cyberknife", "srs"],
    "tx_percutaneous": ["radiofrequency", "rhizotomy", "balloon", "glycerol"],
}


def outcome_from_row(row):
    v = _s(row.get("treatment_outcome"))
    if v in ("complete_relief", "partial_relief", "no_relief"):
        return v
    if v in ("", NR):
        v = _s(row.get("outcome_detail"))        # v2 files carry detail text separately
        if v in ("", NR):
            return NR
    if re.search(r"no (pain )?relief|no response|failed|refractory|no improvement", v):
        return "no_relief"
    if re.search(r"partial|improv|reduc|some relief", v):
        return "partial_relief"
    if re.search(r"complete|pain[- ]free|resolv|full|total|excellent|good|relief", v):
        return "complete_relief"
    return "unmapped:" + v[:30]


def normalize(raw):
    out = pd.DataFrame(index=raw.index)
    out["case_id"] = raw["case_id"] if "case_id" in raw else raw["pmid"].astype(str) + "_" + raw.groupby("pmid").cumcount().astype(str)
    out["pmid"] = raw["pmid"].astype(str)
    out["patient_age"] = raw["patient_age"].apply(parse_age)
    out["symptom_duration_months"] = raw["symptom_duration"].apply(parse_duration_months)
    out["sex"] = raw["sex"].apply(lambda x: pick(x, {"male": "male", "m": "male", "female": "female", "f": "female"}))
    out["laterality"] = raw["laterality"].apply(lambda x: pick(
        x, {"left": "left", "right": "right", "bilateral": "bilateral"})) if "laterality" in raw else NR
    out["pain_character"] = raw["pain_character"].apply(lambda x: pick(
        x, {"paroxysmal": "paroxysmal", "mixed": "mixed", "continuous": "continuous"}))

    branch = raw["branch_involved"].apply(_s)
    for b, name in (("1", "ophthalmic"), ("2", "maxillary"), ("3", "mandibular")):
        out[f"branch_V{b}"] = branch.apply(lambda s, b=b, name=name: int(_branch_has(s, int(b), name)))

    trig = raw["trigger_factors"].apply(_s)
    for col, kws in TRIGGER_KW.items():
        out[col] = [int(any(k in t for k in kws)) for t in trig]
    out["trig_other"] = [int(t not in ("", NR, "none", "none identified") and not any(any(k in t for k in kws) for kws in TRIGGER_KW.values())) for t in trig]
    out["triggers_reported"] = [int(t not in ("", NR)) for t in trig]

    out["MRI_finding"] = raw["MRI_finding"].apply(lambda x: pick(x, {
        "compression_yes": "compression_yes", "compression_no": "compression_no", "other_lesion": "other_lesion"}))
    vessel_map = {"superior_cerebellar_artery": "superior_cerebellar_artery", "aica": "aica", "vein": "vein",
                  "other": "other", "none": "none"}
    out["offending_vessel"] = raw["offending_vessel"].apply(lambda x: _vessel(x, vessel_map))
    for col, default in (("nerve_morphological_change", NR), ("underlying_cause", NR)):
        out[col] = raw[col].apply(lambda x: _s(x).replace(" ", "_") or NR) if col in raw else default

    tx = raw["treatment_given"].apply(_s)
    txm = [multi_hot(t, TX_KW) for t in tx]
    for col in TX_KW:
        out[col] = [m[col] for m in txm]
    out["tx_other"] = [int(t not in ("", NR, "none") and not any(m.values())) for t, m in zip(tx, txm)]
    out["treatment_reported"] = [int(t not in ("", NR)) for t in tx]

    out[TARGET_CLASS] = raw["TN_classification"].apply(
        lambda x: {"classic": "classic", "classical": "classic", "idiopathic": "idiopathic", "secondary": "secondary"}.get(_s(x), NR))
    out[TARGET_OUTCOME] = raw.apply(outcome_from_row, axis=1)

    out = out[CLEAN_COLUMNS]
    _unmapped_report(out)
    return out


def _vessel(x, mapping):
    s = _s(x).replace(" ", "_")
    if s in mapping:
        return mapping[s]
    if s in ("", NR):
        return NR
    if "superior" in s or s == "sca":
        return "superior_cerebellar_artery"
    if "anterior_inferior" in s:
        return "aica"
    if "vein" in s or "venous" in s:
        return "vein"
    return "other"


def _unmapped_report(clean):
    print("--- unmapped_report (anything starting 'unmapped:' needs a keyword/mapping) ---")
    bad = 0
    for c in clean.columns:
        if clean[c].dtype == object:
            vals = clean[c][clean[c].astype(str).str.startswith("unmapped:")]
            if len(vals):
                bad += len(vals)
                print(f"{c}: {vals.value_counts().head(10).to_dict()}")
    print("none" if bad == 0 else f"{bad} unmapped cells")


def load_and_normalize(path):
    return normalize(pd.read_csv(path, dtype={"pmid": str}))


if __name__ == "__main__":
    import sys
    df = load_and_normalize(sys.argv[1])
    df.to_csv("clean_dataset.csv", index=False)
    print(df.shape, "-> clean_dataset.csv")
