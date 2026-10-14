# A5 (v2) - run LLM extraction over the filtered corpus.
#
# Design notes:
#   - case_id per patient (pmid + index); Track B splits train/test by paper
#   - checkpointing: every answer is appended to extraction_raw.jsonl; a rerun skips
#     finished PMIDs and retries failed ones
#   - bad / non-dict answers do not crash the run; problems go in the `issues` column
#   - aggregate series (>10 patients) go to aggregate_series.csv
#   - timeouts, retry only when retrying can help, longer backoff
#   - MULTI-PROVIDER with automatic failover (OpenRouter -> Groq -> ...). Built for free
#     tiers: when one provider hits its limit it is put on "cooldown" and the next one
#     takes over; if all are exhausted the run stops cleanly and a rerun continues.
#   - provider + model are stored per row. Mixing models inside one dataset is a
#     methodological problem, so the script prints a warning when it happens.
#
# SETTINGS (environment variables; PowerShell: $env:NAME = "value")
#   LLM_PROVIDERS          order to try, default "openrouter,groq"   (also: anthropic)
#   OPENROUTER_API_KEY, OPENROUTER_MODEL
#   GROQ_API_KEY,       GROQ_MODEL
#   ANTHROPIC_API_KEY,  ANTHROPIC_MODEL
#   LLM_SLEEP_SECONDS      pause after each paper, default 3 (free tiers cap requests/minute)
#   LLM_COOLDOWN_SECONDS   how long a rate-limited provider is skipped, default 900
# There is NO default model name on purpose: free model ids change often. You choose.

import os
import re
import json
import time
import requests
import pandas as pd

from a4_extraction_prompt import (
    build_extraction_prompt, SYSTEM_PROMPT, EXTRACTION_FIELDS, SCHEMA_VERSION, validate_record,
)

RAW_LOG = "extraction_raw.jsonl"
MAX_TOKENS = 4000   # multi-patient answers need room; the original 1000 could truncate
TEMPERATURE = 0     # reproducibility; set to None below if a model rejects it
SLEEP_S = float(os.environ.get("LLM_SLEEP_SECONDS", "3"))
COOLDOWN_S = int(os.environ.get("LLM_COOLDOWN_SECONDS", "900"))

PROVIDER_SPECS = {
    "openrouter": dict(url="https://openrouter.ai/api/v1/chat/completions",
                       key_env="OPENROUTER_API_KEY", model_env="OPENROUTER_MODEL", style="openai"),
    "groq": dict(url="https://api.groq.com/openai/v1/chat/completions",
                 key_env="GROQ_API_KEY", model_env="GROQ_MODEL", style="openai"),
    "anthropic": dict(url="https://api.anthropic.com/v1/messages",
                      key_env="ANTHROPIC_API_KEY", model_env="ANTHROPIC_MODEL", style="anthropic"),
}


class RetryableError(Exception):
    """Rate limit / server busy / empty reply: waiting or switching provider may help."""
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class ProviderDown(Exception):
    """Bad key, no credit, unknown model: this provider cannot work for the rest of the run."""


class AllProvidersBusy(Exception):
    """Every provider is disabled or cooling down. Stop cleanly; progress is already saved."""


# ----------------------------------------------------------------------------- providers
def load_providers(env=None):
    env = os.environ if env is None else env
    names = [n.strip().lower() for n in env.get("LLM_PROVIDERS", "openrouter,groq").split(",") if n.strip()]
    providers = []
    for n in names:
        spec = PROVIDER_SPECS.get(n)
        if spec is None:
            raise SystemExit(f"unknown provider '{n}'. Known: {', '.join(PROVIDER_SPECS)}")
        key, model = env.get(spec["key_env"]), env.get(spec["model_env"])
        if not key or not model:
            print(f"  skipping {n}: set both {spec['key_env']} and {spec['model_env']} to use it")
            continue
        providers.append(dict(name=n, url=spec["url"], style=spec["style"], key=key, model=model,
                              disabled=False, cooldown_until=0.0))
    if not providers:
        raise SystemExit(
            "no usable provider. Set e.g. OPENROUTER_API_KEY + OPENROUTER_MODEL "
            "(and optionally GROQ_API_KEY + GROQ_MODEL).")
    print("providers in order:", ", ".join(f"{p['name']} ({p['model']})" for p in providers))
    return providers


def _retry_after(resp):
    try:
        return float(resp.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def call_provider(p, prompt):
    """One HTTP call -> raw text. Separated out so tests can fake it."""
    if p["style"] == "anthropic":
        headers = {"x-api-key": p["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": p["model"], "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}]}
    else:  # "openai" style: OpenRouter, Groq
        headers = {"Authorization": f"Bearer {p['key']}", "content-type": "application/json"}
        body = {"model": p["model"], "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}]}
    if TEMPERATURE is not None:
        body["temperature"] = TEMPERATURE

    r = requests.post(p["url"], headers=headers, json=body, timeout=120)
    if r.status_code in (401, 402, 403, 404):
        raise ProviderDown(f"{p['name']}: status {r.status_code} ({r.text[:150]})")
    if r.status_code in (408, 425, 429, 500, 502, 503, 504, 529):
        raise RetryableError(f"{p['name']}: status {r.status_code}", _retry_after(r))
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):          # some gateways send errors with status 200
        raise RetryableError(f"{p['name']}: {str(data['error'])[:150]}")
    if p["style"] == "anthropic":
        return "".join(b["text"] for b in data["content"] if b["type"] == "text")
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not content:
        raise RetryableError(f"{p['name']}: empty reply")
    return content


# ----------------------------------------------------------------------------- parsing
def parse_json_text(text):
    """Tolerant: strips <think> blocks and code fences, then falls back to the first JSON value in the text."""
    if not text or not text.strip():
        raise json.JSONDecodeError("empty reply", "", 0)
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as first_error:
        dec = json.JSONDecoder()
        for i, ch in enumerate(t):
            if ch in "[{":
                try:
                    return dec.raw_decode(t[i:])[0]
                except json.JSONDecodeError:
                    continue
        raise first_error


# ----------------------------------------------------------------------------- one paper
def _available(p):
    return not p["disabled"] and time.time() >= p["cooldown_until"]


def extract_one(case_text, providers, retries=3):
    """Try providers in order. Returns (result|None, raw|None, provider|None).
    Raises AllProvidersBusy when no provider can take the request right now."""
    prompt = build_extraction_prompt(case_text)
    last_raw = None
    for p in providers:
        if not _available(p):
            continue
        for attempt in range(retries):
            try:
                last_raw = call_provider(p, prompt)
                return parse_json_text(last_raw), last_raw, p
            except json.JSONDecodeError as e:
                print(f"  {p['name']} attempt {attempt+1}: bad JSON ({e})")
                time.sleep(1)
            except RetryableError as e:
                wait = e.retry_after if e.retry_after is not None else 5 * 2 ** attempt
                print(f"  {p['name']} attempt {attempt+1}: {e}")
                if attempt == retries - 1 or wait > 60:
                    p["cooldown_until"] = time.time() + max(COOLDOWN_S, wait)
                    print(f"  {p['name']} on cooldown for {int(max(COOLDOWN_S, wait))}s -> switching provider")
                    break
                time.sleep(wait)
            except ProviderDown as e:
                p["disabled"] = True
                print(f"  {e} -> disabled for this run")
                break
            except requests.RequestException as e:
                print(f"  {p['name']} attempt {attempt+1}: network error {e}")
                time.sleep(2 ** attempt)
    if not any(_available(p) for p in providers):
        raise AllProvidersBusy()
    print("  gave up on this one (every available provider returned unreadable output)")
    return None, last_raw, None


# ----------------------------------------------------------------------------- log -> tables
def load_done(path=RAW_LOG):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[str(rec["pmid"])] = rec          # later lines override earlier ones
    return done


def records_from_log(done, df):
    """Turn the raw log into (patient rows, aggregate rows, failed pmids) for the papers in df."""
    titles = dict(zip(df["pmid"].astype(str), df["title"]))
    rows, aggregates, failed = [], [], []
    for pmid in df["pmid"].astype(str):
        rec = done.get(pmid)
        if rec is None:
            continue                                      # not attempted yet
        result = rec.get("result")
        if result is None:
            failed.append(pmid)
            continue
        items = result if isinstance(result, list) else [result]
        k = 0
        for p in items:
            if isinstance(p, dict) and p.get("aggregate_series"):
                aggregates.append({"pmid": pmid, "title": titles.get(pmid, ""),
                                   "n_patients": p.get("n_patients", "not_reported")})
                continue
            row = {"case_id": f"{pmid}_{k}", "pmid": pmid, "title": titles.get(pmid, "")}
            issues = validate_record(p)
            for field in EXTRACTION_FIELDS:
                row[field] = p.get(field, "not_reported") if isinstance(p, dict) else "not_reported"
            row["issues"] = ";".join(issues)
            row["schema_version"] = SCHEMA_VERSION
            row["provider"] = rec.get("provider", "")
            row["model"] = rec.get("model", "")
            rows.append(row)
            k += 1
    return pd.DataFrame(rows), pd.DataFrame(aggregates), failed


def run_extraction(df, providers, sleep_s=None):
    sleep_s = SLEEP_S if sleep_s is None else sleep_s
    done = load_done()
    todo = [r for _, r in df.iterrows() if done.get(str(r["pmid"]), {}).get("result") is None]
    print(f"{len(df) - len(todo)} already done, {len(todo)} to do")

    with open(RAW_LOG, "a", encoding="utf-8") as log:
        for i, row in enumerate(todo, 1):
            pmid = str(row["pmid"])
            print(f"[{i}/{len(todo)}] pmid {pmid}")
            try:
                result, raw, used = extract_one(row["abstract"], providers)
            except AllProvidersBusy:
                print("\nAll providers are rate-limited or unusable right now. "
                      "Progress is saved; run the script again later to continue.")
                break
            rec = {"pmid": pmid, "provider": used["name"] if used else "", "model": used["model"] if used else "",
                   "schema_version": SCHEMA_VERSION, "result": result, "raw": raw}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done[pmid] = rec
            time.sleep(sleep_s)

    rows, aggregates, failed = records_from_log(done, df)
    if len(rows):
        counts = rows.groupby(["provider", "model"]).size()
        print("\nrows per provider/model:\n" + counts.to_string())
        if len(counts) > 1:
            print("WARNING: this dataset mixes several models. Report that in the paper, or re-run "
                  "the rows from the weaker model with the stronger one, so one model produced all rows.")
    return rows, aggregates, failed


if __name__ == "__main__":
    providers = load_providers()
    df = pd.read_csv("abstracts_filtered.csv", dtype={"pmid": str})
    out, aggregates, failed = run_extraction(df, providers)
    out.to_csv("extracted_structured.csv", index=False)
    aggregates.to_csv("aggregate_series.csv", index=False)
    if failed:
        open("failed_extractions.txt", "w").write("\n".join(failed))
        print(f"{len(failed)} pmids failed -> failed_extractions.txt (rerun to retry them)")
    n_issue = int((out["issues"] != "").sum()) if len(out) else 0
    print(f"saved {len(out)} patient rows ({n_issue} with vocab issues) to extracted_structured.csv")
