# A7 - how accurate is the AI extraction? Compare it with the hand-filled gold standard.
#
#   python a7_compare_to_gold.py ai    extracted_structured.csv gold.csv
#   python a7_compare_to_gold.py kappa goldA.csv goldB.csv       # do two annotators agree?
#
# For every field and every gold case, the AI's answer lands in one of five boxes:
#   correct            both give a value and they match
#   wrong_value        both give a value but they differ
#   missed             gold has a value, the AI said not_reported
#   invented           gold says not_reported, the AI gave a value   <- the dangerous one
#   both_not_reported  both say not_reported
# Free-text fields (trigger_factors, treatment_given, outcome_detail) are not auto-scored:
# they are listed side by side in a7_free_text_review.csv for a human to read.
# Cases are matched on case_id (pmid_k, k = order of the patient in the text).

import re
import sys
import argparse
import numpy as np
import pandas as pd
from a4_extraction_prompt import VOCAB, EXTRACTION_FIELDS

NR = "not_reported"
FREE_TEXT = {"trigger_factors", "treatment_given", "outcome_detail"}


def is_nr(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip().lower() in ("", NR, "nan", "not reported")


def vocab_norm(x):
    return NR if is_nr(x) else str(x).strip().lower().replace(" ", "_")


def branch_set(x):
    if is_nr(x):
        return frozenset()
    s = str(x).lower()
    out = set()
    for a, b in re.findall(r"v\s*([123])\s*(?:-|–|to)\s*v?\s*([123])", s):
        out |= set(range(int(a), int(b) + 1))
    out |= {int(d) for d in re.findall(r"\bv\s*([123])\b", s)}
    for n, w in ((1, "ophthalmic"), (2, "maxillary"), (3, "mandibular")):
        if w in s:
            out.add(n)
    return frozenset(out)


def duration_months(x):
    if is_nr(x):
        return np.nan
    s = str(x).lower()
    mult = {"year": 12, "yr": 12, "month": 1, "mo": 1, "week": 12 / 52, "wk": 12 / 52, "day": 12 / 365}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(year|yr|month|mo|week|wk|day)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * mult[m.group(3)]
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(year|yr|month|mo|week|wk|day)", s)
    return sum(float(n) * mult[u] for n, u in parts) if parts else np.nan


def same_value(field, g, a):
    """Both already known to be non-empty."""
    if field == "patient_age":
        try:
            return int(float(g)) == int(float(a))
        except (TypeError, ValueError):
            return False
    if field == "branch_involved":
        return branch_set(g) == branch_set(a) and len(branch_set(g)) > 0
    if field == "symptom_duration":
        gm, am = duration_months(g), duration_months(a)
        if np.isnan(gm) or np.isnan(am):
            return str(g).strip().lower() == str(a).strip().lower()
        return abs(gm - am) <= max(1.0, 0.1 * max(gm, am))        # within 1 month or 10 %
    return vocab_norm(g) == vocab_norm(a)


def effectively_nr(field, x):
    if field == "branch_involved":
        return is_nr(x) or len(branch_set(x)) == 0
    if field == "symptom_duration":
        return is_nr(x) or np.isnan(duration_months(x))
    return is_nr(x)


def box(field, g, a):
    gn, an = effectively_nr(field, g), effectively_nr(field, a)
    if gn and an:
        return "both_not_reported"
    if gn:
        return "invented"
    if an:
        return "missed"
    return "correct" if same_value(field, g, a) else "wrong_value"


def compare(ai, gold):
    ai = ai.copy(); gold = gold.copy()
    ai["case_id"] = ai["case_id"].astype(str); gold["case_id"] = gold["case_id"].astype(str)
    ai = ai.drop_duplicates("case_id").set_index("case_id")
    gold = gold.drop_duplicates("case_id").set_index("case_id")
    no_ai = [c for c in gold.index if c not in ai.index]
    both = [c for c in gold.index if c in ai.index]

    rows, details, free = [], [], []
    for f in EXTRACTION_FIELDS:
        if f in FREE_TEXT:
            for c in both:
                free.append(dict(case_id=c, field=f, gold=gold.at[c, f], ai=ai.at[c, f]))
            continue
        counts = {"correct": 0, "wrong_value": 0, "missed": 0, "invented": 0, "both_not_reported": 0}
        for c in both:
            b = box(f, gold.at[c, f], ai.at[c, f])
            counts[b] += 1
            if b in ("wrong_value", "missed", "invented"):
                details.append(dict(case_id=c, field=f, box=b, gold=gold.at[c, f], ai=ai.at[c, f]))
        n = sum(counts.values())
        has_val = counts["correct"] + counts["wrong_value"] + counts["missed"]
        gold_silent = counts["invented"] + counts["both_not_reported"]
        rows.append(dict(field=f, n=n, accuracy=(counts["correct"] + counts["both_not_reported"]) / n if n else np.nan,
                         found_when_present=counts["correct"] / has_val if has_val else np.nan,
                         invented_when_silent=counts["invented"] / gold_silent if gold_silent else np.nan,
                         **counts))
    summary = pd.DataFrame(rows).round(3)
    return summary, pd.DataFrame(details), pd.DataFrame(free), no_ai, len(both)


def cohens_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return np.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def annotator_agreement(ga, gb):
    ga = ga.drop_duplicates("case_id").set_index("case_id"); gb = gb.drop_duplicates("case_id").set_index("case_id")
    common = [c for c in ga.index if c in gb.index]
    rows = []
    for f, allowed in VOCAB.items():
        if allowed is None:
            continue
        a = [vocab_norm(ga.at[c, f]) for c in common]; b = [vocab_norm(gb.at[c, f]) for c in common]
        rows.append(dict(field=f, n=len(common), percent_agree=round(float(np.mean([x == y for x, y in zip(a, b)])), 3) if common else np.nan,
                         kappa=round(cohens_kappa(a, b), 3) if common else np.nan))
    return pd.DataFrame(rows), len(common)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("ai"); p1.add_argument("ai_csv"); p1.add_argument("gold_csv")
    p2 = sub.add_parser("kappa"); p2.add_argument("gold_a"); p2.add_argument("gold_b")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.cmd == "ai":
        ai = pd.read_csv(a.ai_csv, dtype=str); gold = pd.read_csv(a.gold_csv, dtype=str)
        summary, details, free, no_ai, n_both = compare(ai, gold)
        print(f"gold cases: {len(gold)}; matched in AI output: {n_both}; AI produced no row for: {len(no_ai)}")
        if no_ai:
            print("  (check case_id numbering, or the AI skipped/failed these):", no_ai[:10])
        print(summary.to_string(index=False))
        summary.to_csv("a7_per_field_accuracy.csv", index=False)
        details.to_csv("a7_disagreements.csv", index=False)
        free.to_csv("a7_free_text_review.csv", index=False)
        print("\nwrote a7_per_field_accuracy.csv, a7_disagreements.csv, a7_free_text_review.csv")
        print("Read 'invented' first: that is the AI making up facts.")
    else:
        ga = pd.read_csv(a.gold_a, dtype=str); gb = pd.read_csv(a.gold_b, dtype=str)
        res, n = annotator_agreement(ga, gb)
        print(f"cases both annotators filled: {n}")
        print(res.to_string(index=False))
        print("\nkappa guide: above 0.8 very good, 0.6-0.8 good, 0.4-0.6 moderate, below 0.4 weak. "
              "NaN = no variation in that column.")
