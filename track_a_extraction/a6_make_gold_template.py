# A6 - make the blank sheet for hand-filling the "gold standard" (the answer key we use
# to measure how accurate the AI extraction is).
#
#   python a6_make_gold_template.py --n 50 --seed 1
#   python a6_make_gold_template.py --n 50 --seed 1 --oversample-rare extracted_structured.csv
#
# Output: gold_template.csv  (one row per paper; abstract included; all answer columns blank)
#         gold_allowed_values.txt (what is allowed in each column)
#
# HOW TO FILL IN (each annotator works on their OWN copy; never look at the AI's answers):
#   - read the abstract, fill every answer column; use exactly the allowed values; write
#     not_reported when the abstract does not say.
#   - one row per PATIENT. A paper with 2 patients: copy the row, set case_id to <pmid>_0 and
#     <pmid>_1 in the order the patients appear in the text.
#   - TN_classification = the type the AUTHORS state. Do not work it out yourself.
#   - delete rows for papers that are not usable (e.g. a 200-patient series) and say so in notes.
# --oversample-rare uses the AI's answers ONLY to choose which papers to include (so the rarer
# types are represented); the annotators still never see those answers.

import argparse
import pandas as pd
from a4_extraction_prompt import VOCAB, EXTRACTION_FIELDS
import logging

log = logging.getLogger(__name__)


def make_template(abstracts: pd.DataFrame, n: int, seed: int, extracted: pd.DataFrame | None = None) -> pd.DataFrame:
    abstracts = abstracts.copy()
    abstracts["pmid"] = abstracts["pmid"].astype(str)
    n = min(n, len(abstracts))
    chosen = pd.Index([])
    if extracted is not None and len(extracted):
        ex = extracted.copy()
        ex["pmid"] = ex["pmid"].astype(str)
        rare = ex[(ex["TN_classification"].isin(["secondary", "idiopathic"]))
                  | (ex["MRI_finding"] == "other_lesion")]["pmid"].unique()
        rare_pool = abstracts[abstracts["pmid"].isin(rare)]
        take = min(len(rare_pool), n // 2)
        chosen = rare_pool.sample(take, random_state=seed).index
    rest = abstracts.drop(chosen)
    chosen = chosen.append(rest.sample(n - len(chosen), random_state=seed).index)
    sheet = abstracts.loc[chosen, ["pmid", "title", "abstract"]].reset_index(drop=True)
    sheet.insert(0, "case_id", sheet["pmid"] + "_0")
    for f in EXTRACTION_FIELDS:
        sheet[f] = ""
    sheet["annotator"] = ""
    sheet["notes"] = ""
    return sheet.sample(frac=1, random_state=seed + 1).reset_index(drop=True)   # shuffled order


def allowed_values_text() -> str:
    lines = ["ALLOWED VALUES (write not_reported when the abstract does not say)", ""]
    for f, allowed in VOCAB.items():
        lines.append(f"{f}: " + (" | ".join(allowed) if allowed else "free text (patient_age = whole number of years)"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstracts", default="abstracts_filtered.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--oversample-rare", metavar="EXTRACTED_CSV", default=None)
    a = ap.parse_args()
    ab = pd.read_csv(a.abstracts, dtype={"pmid": str})
    ex = pd.read_csv(a.oversample_rare, dtype={"pmid": str}) if a.oversample_rare else None
    sheet = make_template(ab, a.n, a.seed, ex)
    sheet.to_csv("gold_template.csv", index=False)
    open("gold_allowed_values.txt", "w", encoding="utf-8").write(allowed_values_text())
    log.info(f"wrote gold_template.csv ({len(sheet)} papers) and gold_allowed_values.txt")
    log.info("Give each annotator their own copy. Do not show them the AI output.")
