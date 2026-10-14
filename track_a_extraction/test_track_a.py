# Offline tests for the Track A scripts (no network, no API key).
# run: python test_track_a.py
import json, os, tempfile
import pandas as pd

import a2_fetch_abstracts as a2
import a3_filter_dedupe as a3
import a4_extraction_prompt as a4
import a5_run_extraction as a5
import a6_make_gold_template as a6
import a7_compare_to_gold as a7

XML = b"""<?xml version="1.0"?><PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
 <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>J Test</Title></Journal>
 <ArticleTitle>Trigeminal neuralgia: a case report</ArticleTitle>
 <Abstract><AbstractText Label="CASE">A 50-year-old woman with paroxysmal V2 pain and MRI compression by the SCA, treated with carbamazepine and then microvascular decompression.</AbstractText></Abstract>
 <Language>eng</Language>
 <PublicationTypeList><PublicationType>Case Reports</PublicationType><PublicationType>Journal Article</PublicationType></PublicationTypeList>
</Article><CommentsCorrectionsList><CommentsCorrections><PMID>999</PMID></CommentsCorrections></CommentsCorrectionsList></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_a2_parse():
    r = a2.parse_pubmed_xml(XML)[0]
    assert r["pmid"] == "111", r            # not the 999 inside CommentsCorrections
    assert r["year"] == "2019", r           # from MedlineDate
    assert "Case Reports" in r["publication_types"]
    assert r["language"] == "eng"


def test_a4_prompt_is_valid_json_and_examples_validate():
    p = a4.build_extraction_prompt("x")
    assert "'patient_age'" not in p, "examples must be json.dumps, not python repr"
    assert '"patient_age": "62"' in p
    for ex in a4.FEW_SHOT_EXAMPLES:
        for rec in (ex["json"] if isinstance(ex["json"], list) else [ex["json"]]):
            assert a4.validate_record(rec) == []
    assert "bad_value:sex='M'" in a4.validate_record({**a4.FEW_SHOT_EXAMPLES[0]["json"], "sex": "M"})


def long_abs(extra):
    return ("A 50-year-old patient presented with facial pain. " + extra + " " + "padding " * 30)


def test_a3_filters():
    rows = [
        dict(pmid="1", title="TN case", abstract=long_abs("paroxysmal pain V2."), year="2020", publication_types="Case Reports", language="eng"),
        dict(pmid="1", title="TN case dup pmid", abstract=long_abs("paroxysmal pain."), year="2020", publication_types="Case Reports", language="eng"),
        dict(pmid="2", title="TN case", abstract=long_abs("paroxysmal pain."), year="2020", publication_types="Case Reports", language="eng"),   # same title+year
        dict(pmid="3", title="TN case", abstract=long_abs("paroxysmal pain."), year="2021", publication_types="Case Reports", language="eng"),   # same title, other year: keep
        dict(pmid="4", title="Nav1.7 only", abstract=long_abs("A Nav1.7 variant was found."), year="2020", publication_types="Case Reports", language="eng"),  # 'v1' inside Nav1.7 must NOT match
        dict(pmid="5", title="Review", abstract=long_abs("paroxysmal pain."), year="2020", publication_types="Review", language="eng"),
        dict(pmid="6", title="Case + lit review", abstract=long_abs("paroxysmal pain. We also did a systematic review."), year="2020", publication_types="Case Reports|Review", language="eng"),
        dict(pmid="7", title="Short", abstract="paroxysmal", year="2020", publication_types="Case Reports", language="eng"),
        dict(pmid="8", title="Francais", abstract=long_abs("paroxysmal pain."), year="2020", publication_types="Case Reports", language="fre"),
    ]
    kept, excluded = a3.apply_filters(pd.DataFrame(rows))
    assert set(kept["pmid"]) == {"1", "3", "6"}, set(kept["pmid"])
    reasons = dict(zip(excluded["pmid"] + "|" + excluded["title"], excluded["exclude_reason"]))
    assert reasons["4|Nav1.7 only"] == "no_relevant_term"
    assert bool(kept.loc[kept.pmid == "6", "maybe_review_wording"].iloc[0])


def prov(name, model="m"):
    return dict(name=name, url="", style="openai", key="k", model=model, disabled=False, cooldown_until=0.0)


def fresh_dir():
    d = tempfile.mkdtemp(); os.chdir(d)
    a5.time.sleep = lambda s: None


def test_a5_parse_json_text():
    obj = {"a": 1}
    assert a5.parse_json_text('{"a": 1}') == obj
    assert a5.parse_json_text('```json\n{"a": 1}\n```') == obj
    assert a5.parse_json_text('<think>hmm {not json}</think>\n{"a": 1}') == obj
    assert a5.parse_json_text('Sure! Here it is: [{"a": 1}, {"a": 2}] hope that helps') == [{"a": 1}, {"a": 2}]
    for bad in ("", "no json here", "<think>only thinking</think>"):
        try:
            a5.parse_json_text(bad); raise AssertionError("should fail: " + bad)
        except json.JSONDecodeError:
            pass


def test_a5_pipeline_with_fake_api():
    fresh_dir()
    ok_obj = dict(a4.FEW_SHOT_EXAMPLES[0]["json"])
    answers = {
        "A": json.dumps(ok_obj),                                                   # single patient
        "B": "```json\n" + json.dumps(a4.FEW_SHOT_EXAMPLES[2]["json"]) + "\n```",  # fenced, 2 patients
        "C": json.dumps({"aggregate_series": True, "n_patients": 120}),
        "D": "not json at all",                                                    # must fail, not crash
        "E": json.dumps(["oops", dict(ok_obj, sex="M")]),                          # non-dict + bad vocab
    }
    calls = []
    def fake(p, prompt):
        calls.append(1)
        return answers[prompt.rsplit("Text:", 1)[1].strip().split("\n")[0]]
    a5.call_provider = fake

    df = pd.DataFrame({"pmid": list("ABCDE"), "title": ["t"] * 5, "abstract": list("ABCDE")})
    out, agg, failed = a5.run_extraction(df, [prov("fake")], sleep_s=0)
    assert failed == ["D"], failed
    assert list(agg["n_patients"]) == [120]
    assert set(out["case_id"]) == {"A_0", "B_0", "B_1", "E_0", "E_1"}, set(out["case_id"])
    e = out[out.pmid == "E"].set_index("case_id")["issues"]
    assert e["E_0"] == "not_a_dict" and "bad_value:sex" in e["E_1"], e
    assert set(out["provider"]) == {"fake"}

    n_before = len(calls)
    out2, _, failed2 = a5.run_extraction(df, [prov("fake")], sleep_s=0)   # rerun: only failed D is retried
    assert len(calls) - n_before >= 1 and failed2 == ["D"] and len(out2) == len(out)


def test_a5_failover_primary_rate_limited():
    fresh_dir()
    good = json.dumps(a4.FEW_SHOT_EXAMPLES[0]["json"])
    seen = {"primary": 0, "secondary": 0}
    def fake(p, prompt):
        seen[p["name"]] += 1
        if p["name"] == "primary":
            raise a5.RetryableError("429")
        return good
    a5.call_provider = fake
    df = pd.DataFrame({"pmid": list("ABCD"), "title": ["t"] * 4, "abstract": list("ABCD")})
    primary, secondary = prov("primary", "m1"), prov("secondary", "m2")
    out, _, failed = a5.run_extraction(df, [primary, secondary], sleep_s=0)
    assert len(out) == 4 and not failed
    assert set(out["provider"]) == {"secondary"} and set(out["model"]) == {"m2"}
    assert seen["primary"] == 3, seen        # tried (3 retries) once, then skipped while on cooldown
    assert primary["cooldown_until"] > 0


def test_a5_provider_down_is_disabled_and_next_used():
    fresh_dir()
    good = json.dumps(a4.FEW_SHOT_EXAMPLES[0]["json"])
    def fake(p, prompt):
        if p["name"] == "primary":
            raise a5.ProviderDown("401 bad key")
        return good
    a5.call_provider = fake
    df = pd.DataFrame({"pmid": ["A", "B"], "title": ["t"] * 2, "abstract": ["A", "B"]})
    primary = prov("primary")
    out, _, _ = a5.run_extraction(df, [primary, prov("secondary")], sleep_s=0)
    assert len(out) == 2 and primary["disabled"]


def test_a5_all_busy_stops_cleanly_and_resumes_later():
    fresh_dir()
    good = json.dumps(a4.FEW_SHOT_EXAMPLES[0]["json"])
    state = {"busy": False}
    def fake(p, prompt):
        text = prompt.rsplit("Text:", 1)[1].strip().split("\n")[0]
        if text in ("C", "D", "E") or state["busy"]:
            raise a5.RetryableError("429")
        return good
    a5.call_provider = fake
    df = pd.DataFrame({"pmid": list("ABCDE"), "title": ["t"] * 5, "abstract": list("ABCDE")})
    out, _, failed = a5.run_extraction(df, [prov("p1"), prov("p2")], sleep_s=0)
    assert set(out["pmid"]) == {"A", "B"} and failed == []        # C was never recorded as failed
    lines = [json.loads(l) for l in open(a5.RAW_LOG)]
    assert [l["pmid"] for l in lines] == ["A", "B"]

    a5.call_provider = lambda p, prompt: good                       # quota is back
    out2, _, failed2 = a5.run_extraction(df, [prov("p1"), prov("p2")], sleep_s=0)
    assert set(out2["pmid"]) == set("ABCDE") and failed2 == []


def test_a5_load_providers():
    env = {"LLM_PROVIDERS": "openrouter,groq", "OPENROUTER_API_KEY": "k1", "OPENROUTER_MODEL": "m1",
           "GROQ_API_KEY": "k2"}                                      # groq has no model -> skipped
    assert [p["name"] for p in a5.load_providers(env)] == ["openrouter"]
    env["GROQ_MODEL"] = "m2"
    assert [p["name"] for p in a5.load_providers(env)] == ["openrouter", "groq"]
    for bad in ({}, {"LLM_PROVIDERS": "nonsense"}):
        try:
            a5.load_providers(bad); raise AssertionError("should exit")
        except SystemExit:
            pass


def frame(rows):
    base = {f: "not_reported" for f in a4.EXTRACTION_FIELDS}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_a6_gold_template_blind_and_covers_rare():
    ab = pd.DataFrame({"pmid": [str(i) for i in range(20)], "title": ["t"] * 20, "abstract": ["abs"] * 20})
    ex = pd.DataFrame({"pmid": ["3", "4", "5", "6"], "TN_classification": ["secondary", "idiopathic", "classic", "classic"],
                       "MRI_finding": ["compression_no"] * 4})
    sheet = a6.make_template(ab, 10, 1, ex)
    assert len(sheet) == 10 and sheet["pmid"].nunique() == 10
    assert set(["3", "4"]) <= set(sheet["pmid"]), "rare cases should be oversampled"
    assert (sheet["case_id"] == sheet["pmid"] + "_0").all()
    assert (sheet[a4.EXTRACTION_FIELDS] == "").all().all(), "answer columns must be blank (annotators must not see AI output)"
    assert "TN_classification: classic | idiopathic | secondary | not_reported" in a6.allowed_values_text()


def test_a7_boxes_and_parsers():
    gold = frame([
        dict(case_id="a_0", sex="female", patient_age="62", branch_involved="V2, V3", symptom_duration="3 years", MRI_finding="compression_yes"),
        dict(case_id="b_0", sex="male", patient_age="40", branch_involved="V1-V3", symptom_duration="2 years", MRI_finding="compression_no"),
        dict(case_id="c_0", sex="female", branch_involved="V1"),
        dict(case_id="d_0"),
        dict(case_id="e_0", sex="male"),                       # AI will have no row for this one
    ])
    ai = frame([
        dict(case_id="a_0", sex="female", patient_age="62.0", branch_involved="V3, V2", symptom_duration="36 months", MRI_finding="compression_yes"),
        dict(case_id="b_0", sex="female", patient_age="41", branch_involved="V1, V2, V3", symptom_duration="30 months", MRI_finding="Compression No"),
        dict(case_id="c_0", sex="not_reported", branch_involved="V2"),
        dict(case_id="d_0", sex="male"),
    ])
    summary, details, free, no_ai, n_both = a7.compare(ai, gold)
    assert no_ai == ["e_0"] and n_both == 4
    s = summary.set_index("field")
    assert s.loc["sex", ["correct", "wrong_value", "missed", "invented"]].tolist() == [1, 1, 1, 1], s.loc["sex"]
    assert s.loc["patient_age", "correct"] == 1 and s.loc["patient_age", "wrong_value"] == 1       # 62 == 62.0 ; 40 != 41
    assert s.loc["branch_involved", "correct"] == 2 and s.loc["branch_involved", "wrong_value"] == 1  # V3,V2 == V2,V3 ; V1-V3 == V1,V2,V3 ; V1 != V2
    assert s.loc["symptom_duration", "correct"] == 1 and s.loc["symptom_duration", "wrong_value"] == 1  # 3 y == 36 mo ; 2 y != 30 mo
    assert s.loc["MRI_finding", "correct"] == 2                                                      # case-insensitive, spaces -> underscores
    assert set(free["field"]) == a7.FREE_TEXT
    assert "invented" in set(details["box"])


def test_a7_kappa():
    assert abs(a7.cohens_kappa(list("xxyy"), list("xyyy")) - 0.5) < 1e-9
    assert a7.cohens_kappa(list("xx"), list("xx")) != a7.cohens_kappa(list("xx"), list("xx"))   # NaN: no variation
    assert a7.cohens_kappa(list("xy"), list("xy")) == 1.0
    ga = frame([dict(case_id="a_0", sex="male"), dict(case_id="b_0", sex="female")])
    res, n = a7.annotator_agreement(ga, ga)
    assert n == 2 and res.set_index("field").loc["sex", "percent_agree"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
