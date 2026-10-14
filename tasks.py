"""Task runner (stdlib only): python tasks.py extract | model | test"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A, B = ROOT / "track_a_extraction", ROOT / "track_b_modeling"


def run(script_or_args: list[str], cwd: Path) -> None:
    subprocess.run([sys.executable, *script_or_args], cwd=cwd, check=True)


def extract() -> None:
    for s in ("a1_pubmed_search.py", "a2_fetch_abstracts.py", "a3_filter_dedupe.py", "a5_run_extraction.py"):
        run([s], A)


def model() -> None:
    run(["b4_run_mock.py"], B)


def test() -> None:
    for d in (A, B):
        run(["-m", "pytest", "-q"], d)


TARGETS = {"extract": extract, "model": model, "test": test}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=TARGETS)
    TARGETS[p.parse_args().target]()
