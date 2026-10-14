# A2 - fetch abstracts + metadata for the PMIDs from A1, and check PMC links.
#
# Design notes:
#   - also saves publication_types + language (a3 uses publication types to drop
#     reviews properly instead of string-matching the abstract)
#   - year falls back to <MedlineDate> (many records have no <Year>)
#   - PMID read from MedlineCitation/PMID (not the first PMID anywhere in the record)
#   - timeouts + retry with backoff on every request (original: one hiccup = lost run)
#   - reports PMIDs that efetch did not return
#   - email read from NCBI_EMAIL env var
#   - "has_full_text" renamed to in_pmc: being in PMC is NOT the same as being
#     open-access / text-minable

import os
import re
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import logging

log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MY_EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com")


def http_get(url, params, retries=4, timeout=60):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.info(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def chunk_list(lst: list, size: int) -> list[list]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def parse_pubmed_xml(content: bytes) -> list[dict]:
    """Parse an efetch XML payload (bytes/str) into a list of record dicts."""
    root = ET.fromstring(content)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext("./MedlineCitation/PMID")

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_bits = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(b.itertext()).strip() for b in abstract_bits)

        journal = article.findtext(".//Journal/Title") or ""

        year = article.findtext(".//JournalIssue/PubDate/Year") or ""
        if not year:
            medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
            m = re.search(r"(19|20)\d{2}", medline_date)
            year = m.group(0) if m else ""

        pub_types = [
            (pt.text or "").strip()
            for pt in article.findall(".//PublicationTypeList/PublicationType")
        ]
        language = article.findtext(".//Language") or ""

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "publication_types": "|".join(pub_types),
            "language": language,
        })
    return records


def fetch_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict]:
    records = []
    for batch in chunk_list(pmids, batch_size):
        params = {
            "db": "pubmed", "id": ",".join(batch),
            "rettype": "abstract", "retmode": "xml", "email": MY_EMAIL,
        }
        r = http_get(EUTILS_BASE + "/efetch.fcgi", params)
        records += parse_pubmed_xml(r.content)
        time.sleep(0.34)

    got = {r["pmid"] for r in records}
    missing = [p for p in pmids if p not in got]
    log.info(f"got {len(records)} records back; {len(missing)} requested PMIDs not returned")
    if missing:
        log.info("  missing (first 10): %s", missing[:10])
    return records


def check_pmc_links(pmids: list[str], batch_size: int = 200) -> dict[str, str]:
    # returns {pmid: pmcid} - empty string if no PMC version found
    result = {p: "" for p in pmids}
    for batch in chunk_list(pmids, batch_size):
        params = {"dbfrom": "pubmed", "db": "pmc", "id": batch, "retmode": "xml", "email": MY_EMAIL}
        r = http_get(EUTILS_BASE + "/elink.fcgi", params)
        root = ET.fromstring(r.content)
        for linkset in root.findall(".//LinkSet"):
            orig_id = linkset.find(".//IdList/Id")
            if orig_id is None:
                continue
            pmc_id = linkset.find(".//LinkSetDb/Link/Id")
            if pmc_id is not None:
                result[orig_id.text] = "PMC" + pmc_id.text
        time.sleep(0.34)

    n_found = len([v for v in result.values() if v])
    log.info(f"{n_found} out of {len(pmids)} have a PMC version (not necessarily open-access)")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open("pmids.txt") as f:
        pmids = [line.strip() for line in f if line.strip()]

    records = fetch_abstracts(pmids)
    pmc_links = check_pmc_links(pmids)
    for rec in records:
        rec["pmcid"] = pmc_links.get(rec["pmid"], "")
        rec["in_pmc"] = bool(rec["pmcid"])

    df = pd.DataFrame(records)
    df.to_csv("abstracts_raw.csv", index=False)
    log.info(f"saved {len(df)} rows to abstracts_raw.csv")
