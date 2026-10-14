# Track A - data extraction (A1-A7)

By Shahd Osama. Each script lists its design notes in its header.

```bash
pip install -r ../requirements.txt
python -m pytest -q               # offline tests, no network / no API key
```

PowerShell (Windows) - not `export`. Free-tier setup: OpenRouter first, Groq as automatic backup:

```powershell
$env:NCBI_EMAIL = "you@example.com"                 # a1/a2: NCBI asks for an e-mail
$env:LLM_PROVIDERS = "openrouter,groq"              # order to try (default); "anthropic" also works
$env:OPENROUTER_API_KEY = "..."
$env:OPENROUTER_MODEL   = "..."                     # you choose; no default on purpose (free model ids change)
$env:GROQ_API_KEY = "..."
$env:GROQ_MODEL   = "..."
$env:LLM_SLEEP_SECONDS = "3"                        # pause per paper; free tiers cap requests/minute
```

How a5 behaves with free tiers: if a provider says "too many requests", it is skipped for 15 minutes and the next one takes over. If every provider is exhausted, the script stops cleanly and prints a message; **just run it again later** - finished papers are skipped. `provider` and `model` are saved on every row. If rows end up from more than one model, the script warns you: report it in the paper, or redo the weaker model's rows with one model.

Run in order (each reads the previous output):

```
python a1_pubmed_search.py --compare   # compare query variants first
python a1_pubmed_search.py             # -> pmids.txt, search_log.json
python a2_fetch_abstracts.py           # -> abstracts_raw.csv
python a3_filter_dedupe.py             # -> abstracts_filtered.csv, excluded.csv (+ flow numbers)
python a5_run_extraction.py            # -> extracted_structured.csv, aggregate_series.csv, extraction_raw.jsonl
```

**Gold standard and accuracy check (A6, A7):**

```
python a6_make_gold_template.py --n 50 --seed 1                    # blank sheet to fill by hand
python a6_make_gold_template.py --n 50 --seed 1 --oversample-rare extracted_structured.csv   # (optional) include more rare types
python a7_compare_to_gold.py ai    extracted_structured.csv gold.csv   # how accurate is the AI?
python a7_compare_to_gold.py kappa goldA.csv goldB.csv                 # do two annotators agree?
```
Annotators never see the AI's answers. `a7` sorts every answer into: correct / wrong_value / missed / **invented** (the AI made it up) / both_not_reported, per field. Free-text fields go to `a7_free_text_review.csv` for a human to read. Rows are matched on `case_id` (`pmid_0`, `pmid_1`, ... in the order patients appear in the text).

**Do a pilot first:** put 20–30 rows in `abstracts_filtered.csv`, run a5, open `extracted_structured.csv`, look at the `issues` column and the share of `not_reported` per field. Delete `extraction_raw.jsonl` before the real run if you change the prompt (a rerun skips PMIDs that already have a result).

Output columns (schema v2): `case_id, pmid, title,` the 15 fields in `a4_extraction_prompt.VOCAB`, `issues, schema_version, provider, model`.

Not tested here (needs network/keys): real calls to NCBI, OpenRouter, Groq, Anthropic - the request/response formats follow each service's OpenAI-style or Anthropic-style API, but check your first real call. Offline tests cover XML parsing, filtering, prompt rendering, and the whole a5 loop with a fake API (single patient, fenced JSON, multi-patient, aggregate series, bad JSON, non-dict answers, resume after failure, provider failover, rate-limit cooldown, bad-key disable, clean stop when every provider is busy).
