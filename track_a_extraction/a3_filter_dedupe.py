# A3 - dedupe + filter the corpus before extraction.
#
# Design notes:
#   - term matching uses word boundaries. Original substring match made "v1" match
#     inside "Nav1.7" (the sodium channel gene) and similar false positives.
#   - reviews are dropped by PubMed PublicationType (from a2), not by finding
#     "systematic review" anywhere in the abstract (which also threw away case reports
#     that include a literature review). Abstract wording only sets a flag.
#   - title-only dedupe was risky ("Trigeminal neuralgia: a case report" is a real,
#     common title). Now title + year; every dropped row is logged.
#   - nothing disappears silently: excluded.csv has a reason per row, and a flow
#     summary is printed (use it for the PRISMA-style numbers in the paper).
#   - non-English abstracts are dropped (the extraction prompt is English).

import re
import pandas as pd

RELEVANT_TERMS = [
    "paroxysmal", "microvascular decompression", "mri", "magnetic resonance",
    "compression", "classic trigeminal", "classical trigeminal",
    "idiopathic trigeminal", "secondary trigeminal",
    "v1", "v2", "v3", "ophthalmic", "maxillary", "mandibular",
    "carbamazepine", "oxcarbazepine", "gabapentin", "gamma knife", "radiosurgery",
    "radiofrequency", "balloon compression",
]
REVIEW_PUBTYPES = {"review", "systematic review", "meta-analysis", "scoping review"}
REVIEW_PHRASES = ["systematic review", "meta-analysis", "narrative review", "literature review"]
MIN_ABSTRACT_LEN = 200

_REL_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in RELEVANT_TERMS) + r")\b", re.I)


def clean_title(title):
    t = re.sub(r"[^a-z0-9\s]", "", str(title).lower())
    return re.sub(r"\s+", " ", t).strip()


def pubtypes(row):
    return {p.strip().lower() for p in str(row.get("publication_types", "")).split("|") if p.strip()}


def apply_filters(df):
    """Return (kept_df, excluded_df). excluded_df has an 'exclude_reason' column."""
    df = df.copy()
    df["exclude_reason"] = ""
    flow = [("input rows", len(df))]

    def mark(mask, reason):
        newly = mask & (df["exclude_reason"] == "")
        df.loc[newly, "exclude_reason"] = reason

    mark(df.duplicated(subset="pmid", keep="first"), "duplicate_pmid")
    flow.append(("after duplicate PMID", (df["exclude_reason"] == "").sum()))

    df["_tc"] = df["title"].apply(clean_title)
    live = df["exclude_reason"] == ""
    dup_title = df[live].duplicated(subset=["_tc", "year"], keep="first")
    mark(df.index.isin(dup_title[dup_title].index), "duplicate_title_year")
    flow.append(("after duplicate title+year", (df["exclude_reason"] == "").sum()))

    if "language" in df.columns:
        mark(~df["language"].fillna("eng").str.lower().isin(["eng", ""]), "non_english")
        flow.append(("after English only", (df["exclude_reason"] == "").sum()))

    abstract = df["abstract"].fillna("")
    mark(abstract.str.len() < MIN_ABSTRACT_LEN, "no_or_short_abstract")
    flow.append(("after abstract >= %d chars" % MIN_ABSTRACT_LEN, (df["exclude_reason"] == "").sum()))

    if "publication_types" in df.columns:
        is_review = df.apply(lambda r: bool(pubtypes(r) & REVIEW_PUBTYPES) and "case reports" not in pubtypes(r), axis=1)
        mark(is_review, "review_publication_type")
        flow.append(("after dropping reviews", (df["exclude_reason"] == "").sum()))

    mark(~abstract.apply(lambda a: bool(_REL_RE.search(a))), "no_relevant_term")
    flow.append(("after relevance terms", (df["exclude_reason"] == "").sum()))

    low = abstract.str.lower()
    df["maybe_review_wording"] = low.apply(lambda a: any(p in a for p in REVIEW_PHRASES))

    print("--- corpus flow ---")
    for name, n in flow:
        print(f"{name:35s} {n}")

    kept = df[df["exclude_reason"] == ""].drop(columns=["_tc", "exclude_reason"]).reset_index(drop=True)
    excluded = df[df["exclude_reason"] != ""].drop(columns=["_tc"])
    print("\nexclusion reasons:\n", excluded["exclude_reason"].value_counts().to_string())
    return kept, excluded


if __name__ == "__main__":
    df = pd.read_csv("abstracts_raw.csv", dtype={"pmid": str})
    kept, excluded = apply_filters(df)
    kept.to_csv("abstracts_filtered.csv", index=False)
    excluded.to_csv("excluded.csv", index=False)
    print(f"\nsaved {len(kept)} rows to abstracts_filtered.csv, {len(excluded)} to excluded.csv")
