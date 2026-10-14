# A4 - prompt + schema for LLM extraction.
#
# Design notes:
#   1. few-shot examples are now rendered with json.dumps (valid JSON). The
#      original used str(dict) -> single quotes -> model copies them -> json.loads fails.
#   2. controlled vocabularies for every categorical field (Track B needs clean values).
#   3. new fields needed to DERIVE the Cruccu classification and to avoid label leakage
#      analysis problems: underlying_cause, nerve_morphological_change, laterality.
#   4. MRI_finding gets "other_lesion" (original example 2 coded an MS plaque as
#      compression_no, losing the information).
#   5. aggregate-series escape hatch (a 200-patient series cannot be 200 JSON objects).
#   6. a validator, so out-of-vocabulary answers are flagged instead of silently kept.
#
# The few-shot texts below are INVENTED illustrations, not real published text.

import json
import logging

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v2"

# field -> allowed values (None = free value, numeric or text, normalised later)
VOCAB = {
    "patient_age": None,                 # integer years, or "not_reported"
    "sex": ["male", "female", "not_reported"],
    "laterality": ["left", "right", "bilateral", "not_reported"],
    "symptom_duration": None,            # keep the author's wording, e.g. "3 years"
    "branch_involved": None,             # any of V1, V2, V3 comma-separated, or "not_reported"
    "pain_character": ["paroxysmal", "mixed", "continuous", "not_reported"],
    "TN_classification": ["classic", "idiopathic", "secondary", "not_reported"],  # AS STATED BY AUTHORS
    "trigger_factors": None,             # free text, comma-separated
    "MRI_finding": ["compression_yes", "compression_no", "other_lesion", "not_reported"],
    "nerve_morphological_change": ["yes", "no", "not_reported"],  # atrophy / dislocation of root
    "offending_vessel": ["superior_cerebellar_artery", "aica", "vein", "other", "none", "not_reported"],
    "underlying_cause": ["none_identified", "multiple_sclerosis", "tumor", "other", "not_reported"],
    "treatment_given": None,             # free text, comma-separated
    "treatment_outcome": ["complete_relief", "partial_relief", "no_relief", "not_reported"],
    "outcome_detail": None,              # free text, e.g. "complete relief at 1-year follow-up"
}

EXTRACTION_FIELDS = list(VOCAB.keys())

SYSTEM_PROMPT = """You are a clinical information extraction assistant. \
You will be given the abstract of a published case report or case series about \
trigeminal neuralgia (TN). Extract the requested fields into JSON.

Rules:
- Only use information explicitly stated in the text. Never guess or infer.
- TN_classification is the label the AUTHORS state. If they do not state one, \
use "not_reported" even if you could work it out from the MRI findings.
- If a field is not reported, use the string "not_reported".
- For fields with an allowed-values list, use EXACTLY one of those values.
- patient_age must be an integer number of years (e.g. "62") or "not_reported".
- If the text describes MULTIPLE individually described patients (up to 10), \
return a JSON array with one object per patient, in the order they appear.
- If the text only reports an aggregate series of MORE than 10 patients without \
per-patient detail, return exactly: {"aggregate_series": true, "n_patients": <integer or "not_reported">}
- Respond with ONLY the JSON (object or array). No prose, no markdown code fences.
"""

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "A 62-year-old woman presented with a 3-year history of "
            "paroxysmal, electric shock-like pain in the right V2 and V3 "
            "distribution, triggered by chewing and light touch. MRI "
            "showed neurovascular compression by the superior cerebellar "
            "artery with atrophy of the nerve root. She was initially treated "
            "with carbamazepine with partial response, and later underwent "
            "microvascular decompression with complete pain relief at 1-year "
            "follow-up. The case was classified as classic trigeminal neuralgia."
        ),
        "json": {
            "patient_age": "62", "sex": "female", "laterality": "right",
            "symptom_duration": "3 years", "branch_involved": "V2, V3",
            "pain_character": "paroxysmal", "TN_classification": "classic",
            "trigger_factors": "chewing, light touch",
            "MRI_finding": "compression_yes", "nerve_morphological_change": "yes",
            "offending_vessel": "superior_cerebellar_artery",
            "underlying_cause": "none_identified",
            "treatment_given": "carbamazepine, microvascular decompression",
            "treatment_outcome": "complete_relief",
            "outcome_detail": "complete pain relief at 1-year follow-up",
        },
    },
    {
        "text": (
            "We report a 34-year-old man with multiple sclerosis who "
            "developed left-sided facial pain with a mixed paroxysmal and "
            "continuous background ache in the V1 territory. No trigger "
            "factors were identified. MRI did not show vascular "
            "compression but did reveal a demyelinating plaque at the "
            "root entry zone. This was classified as secondary trigeminal "
            "neuralgia. He was treated with gabapentin; outcome was not "
            "reported at time of publication."
        ),
        "json": {
            "patient_age": "34", "sex": "male", "laterality": "left",
            "symptom_duration": "not_reported", "branch_involved": "V1",
            "pain_character": "mixed", "TN_classification": "secondary",
            "trigger_factors": "none identified",
            "MRI_finding": "other_lesion", "nerve_morphological_change": "not_reported",
            "offending_vessel": "not_reported",
            "underlying_cause": "multiple_sclerosis",
            "treatment_given": "gabapentin",
            "treatment_outcome": "not_reported", "outcome_detail": "not_reported",
        },
    },
    {   # multi-patient example: returns an ARRAY, and shows several not_reported fields
        "text": (
            "We describe two patients with right-sided facial pain. Patient 1, "
            "a 58-year-old man, had paroxysmal V3 pain for 6 months and an MRI "
            "with no compression or other lesion; he responded fully to "
            "carbamazepine. Patient 2, a 71-year-old woman, had paroxysmal V2 "
            "pain; MRI showed contact with a vein. She declined surgery and "
            "no follow-up was reported."
        ),
        "json": [
            {
                "patient_age": "58", "sex": "male", "laterality": "right",
                "symptom_duration": "6 months", "branch_involved": "V3",
                "pain_character": "paroxysmal", "TN_classification": "not_reported",
                "trigger_factors": "not_reported",
                "MRI_finding": "compression_no", "nerve_morphological_change": "not_reported",
                "offending_vessel": "none", "underlying_cause": "none_identified",
                "treatment_given": "carbamazepine",
                "treatment_outcome": "complete_relief", "outcome_detail": "responded fully to carbamazepine",
            },
            {
                "patient_age": "71", "sex": "female", "laterality": "right",
                "symptom_duration": "not_reported", "branch_involved": "V2",
                "pain_character": "paroxysmal", "TN_classification": "not_reported",
                "trigger_factors": "not_reported",
                "MRI_finding": "compression_yes", "nerve_morphological_change": "not_reported",
                "offending_vessel": "vein", "underlying_cause": "not_reported",
                "treatment_given": "not_reported",
                "treatment_outcome": "not_reported", "outcome_detail": "not_reported",
            },
        ],
    },
]


def build_extraction_prompt(case_text: str) -> str:
    lines = []
    for f, allowed in VOCAB.items():
        lines.append(f"- {f}: " + (" | ".join(allowed) if allowed else "free value"))
    field_list = "\n".join(lines)

    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        # json.dumps (NOT str(dict)) so the model sees real JSON
        examples_text += (
            f"\nExample {i}:\nText: {ex['text']}\nJSON: {json.dumps(ex['json'])}\n"
        )

    return f"""Extract these fields (allowed values after the colon):
{field_list}
{examples_text}
Now extract from this text:
Text: {case_text}
JSON:"""


def validate_record(rec: object) -> list[str]:
    """Return a list of problems for one extracted patient dict (empty list = ok)."""
    problems = []
    if not isinstance(rec, dict):
        return ["not_a_dict"]
    for field, allowed in VOCAB.items():
        val = rec.get(field)
        if val is None:
            problems.append(f"missing:{field}")
        elif allowed is not None and val not in allowed:
            problems.append(f"bad_value:{field}={val!r}")
    age = rec.get("patient_age")
    if age not in (None, "not_reported") and not str(age).strip().isdigit():
        problems.append(f"bad_value:patient_age={age!r}")
    return problems


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info(build_extraction_prompt("A 45-year-old man with V1 pain, paroxysmal, treated with carbamazepine."))
    for ex in FEW_SHOT_EXAMPLES:
        recs = ex["json"] if isinstance(ex["json"], list) else [ex["json"]]
        for r in recs:
            assert validate_record(r) == [], validate_record(r)
    log.info("\nall few-shot examples pass validate_record")
