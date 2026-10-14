# A1 - search PubMed for trigeminal neuralgia case reports/series (esearch).
#
# Design notes:
#   - email from NCBI_EMAIL env var; request timeouts + retry
#   - writes search_log.json (query, run date, total hits, pmids collected): needed for the
#     Methods section and for rerunning the search later
#   - QUERY_VARIANTS: compare hit counts of a few strategies BEFORE committing to one
#     (python a1_pubmed_search.py --compare). Whether to add MeSH/synonyms is the team's call.
# docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/

import os
import sys
import json
import time
import datetime
import requests
import logging

log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MY_EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com")

QUERY = (
    '"trigeminal neuralgia"[Title/Abstract] '
    'AND ("case report"[Publication Type] OR "case reports"[Title/Abstract] '
    'OR "case series"[Title/Abstract])'
)

QUERY_VARIANTS = {
    "original": QUERY,
    "with_mesh_and_synonym": (
        '("trigeminal neuralgia"[Title/Abstract] OR "Trigeminal Neuralgia"[MeSH Terms] '
        'OR "tic douloureux"[Title/Abstract]) '
        'AND ("case report"[Publication Type] OR "case reports"[Title/Abstract] '
        'OR "case series"[Title/Abstract])'
    ),
}


def _get(params, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(EUTILS_BASE + "/esearch.fcgi", params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            return r.json()["esearchresult"]
        except (requests.RequestException, KeyError, ValueError) as e:
            wait = 2 ** attempt
            log.info(f"  esearch failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("esearch failed repeatedly")


def count_hits(query: str) -> int:
    res = _get({"db": "pubmed", "term": query, "retmode": "json", "retmax": 0, "email": MY_EMAIL})
    return int(res["count"])


def search_pubmed_all(query: str = QUERY, page_size: int = 500, max_records: int = 5000) -> tuple[list[str], int]:
    # esearch returns `retmax` ids at a time -> page with retstart.
    # sleep 0.34s: NCBI allows 3 requests/sec without an api key.
    all_pmids, start, total = [], 0, None
    while start < max_records:
        res = _get({"db": "pubmed", "term": query, "retmode": "json",
                    "retmax": page_size, "retstart": start, "email": MY_EMAIL})
        total = int(res["count"])
        ids = res["idlist"]
        if not ids:
            break
        all_pmids += ids
        start += page_size
        time.sleep(0.34)
    if total is not None and total > len(all_pmids):
        log.warning(f"WARNING: {total} hits but only {len(all_pmids)} collected (max_records={max_records})")
    log.info(f"total pmids collected: {len(all_pmids)} (of {total} hits)")
    return all_pmids, total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--compare" in sys.argv:
        for name, q in QUERY_VARIANTS.items():
            log.info(f"{name:25s} {count_hits(q)} hits")
        raise SystemExit
    if MY_EMAIL == "your_email@example.com":
        log.info("note: set NCBI_EMAIL (PowerShell: $env:NCBI_EMAIL='you@x.com') - NCBI asks for it")

    pmids, total = search_pubmed_all(max_records=2000)
    with open("pmids.txt", "w") as f:
        f.write("\n".join(pmids) + "\n")
    with open("search_log.json", "w") as f:
        json.dump({"query": QUERY, "run_date": datetime.date.today().isoformat(),
                   "total_hits": total, "pmids_collected": len(pmids)}, f, indent=2)
    log.info("saved pmids.txt and search_log.json")
