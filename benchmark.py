"""
benchmark.py
============
Four-metric benchmark for TrackMind's Days 6-8 phase.

Metrics match the proposal exactly:
  1. Retrieval Precision    — correct TSI article in top-5 results (target ≥80%)
  2. Cross-Lingual Precision — English query → correct French NNTR article (target ≥70%)
  3. Answer Concordance     — AI verdict matches ground truth on Green-tier queries (target ≥75%)
  4. Confidence Calibration — when AI says ≥90%, is it right ≥90% of the time? (target within 10%)

Ground truth is built from two sources:
  - Retrieval ground truth: which article should appear in top-5 for each query
  - Reasoning ground truth: expected verdict and mandatory citations for each benchmark query

Usage:
    python benchmark.py                  # run full benchmark (requires API)
    python benchmark.py --mock           # run with mock LLM (no API cost)
    python benchmark.py --retrieval-only # run retrieval metrics only
    python benchmark.py --save results.json  # save results to file
"""

import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from retrieval import tri_source_retrieve, measure_retrieval_precision
from reasoning import reason, confidence_gate


# ── Ground truth ──────────────────────────────────────────────────────────────
# Each entry: (query, expected_verdict, required_citations, expected_tsi, expected_nntr)
# required_citations: list of strings that MUST appear in the response citations
# expected_verdict: "CONFLICT DETECTED" / "COMPLIANT" / "INSUFFICIENT DATA"

REASONING_GROUND_TRUTH = [
    # Green-tier queries (both conflicts)
    (
        "Does IberRail's door obstacle detection testing satisfy French RFN requirements under Arrêté 2012 Article 49?",
        "CONFLICT DETECTED",
        ["Art. 49", "4.2.5.5", "NF F31-054"],
        "4.2.5.5",
        "Art. 49",
    ),
    (
        "What CAS single-agent operation requirements does the IB-EMU-450 need to satisfy for French TER services?",
        "CONFLICT DETECTED",
        ["Art. 49", "NF F31-054"],
        "4.2.5.5",
        "Art. 49",
    ),
    (
        "Is IberRail's existing EN 14752 FAT sufficient for French RFN door authorisation?",
        "CONFLICT DETECTED",
        ["Art. 49", "EN 14752", "NF F31-054"],
        "4.2.5.5",
        "Art. 49",
    ),
    (
        "Does the LOC&PAS TSI cover single-agent door operation requirements?",
        "CONFLICT DETECTED",   # TSI is silent → creates gap = conflict with NNTR
        ["Art. 49", "NF F31-054"],
        "4.2.5.5",
        "Art. 49",
    ),
    # Broader compliance queries
    (
        "What are the TSI requirements for exterior passenger access doors?",
        "COMPLIANT",           # pure TSI question — no national rule conflict
        ["4.2.5.5"],
        "4.2.5.5",
        None,
    ),
    (
        "What does Arrêté 2012 Article 49 require for door obstacle detection?",
        "CONFLICT DETECTED",   # asks specifically about French rule vs TSI
        ["Art. 49", "NF F31-054"],
        "4.2.5.5",
        "Art. 49",
    ),
    # Insufficient data query (topic not in ingested documents)
    (
        "What are the Portuguese national rules for door obstacle detection on the IP network?",
        "INSUFFICIENT DATA",   # Portugal NNTR not ingested
        [],
        None,
        None,
    ),
    # Cross-lingual: French query should work
    (
        "Quelles sont les exigences de l'Arrêté 2012 pour les portes d'accès voyageurs?",
        "CONFLICT DETECTED",
        ["Art. 49"],
        None,
        "Art. 49",
    ),
]


# ── Benchmark result dataclasses ──────────────────────────────────────────────

@dataclass
class QueryResult:
    query:           str
    expected_verdict: str
    actual_verdict:   str
    verdict_match:    bool
    expected_citations: list[str]
    actual_citations:   list[str]
    citations_found:    list[str]
    citations_missing:  list[str]
    confidence_tier:    str
    confidence_pct:     int
    is_green_tier:      bool
    concordance_eligible: bool  # Green-tier only
    concordance_pass:   Optional[bool]


@dataclass
class BenchmarkResults:
    # Metric 1: Retrieval Precision
    tsi_precision:        Optional[float]
    nntr_precision:       Optional[float]
    tsi_target_pass:      bool   # ≥80%
    nntr_target_pass:     bool   # ≥70%

    # Metric 2: Cross-Lingual Precision
    cross_lingual_prec:   Optional[float]
    cross_lingual_pass:   bool   # ≥70%

    # Metric 3: Answer Concordance (Green-tier only)
    concordance_n:        int
    concordance_correct:  int
    concordance_rate:     Optional[float]
    concordance_pass:     bool   # ≥75%

    # Metric 4: Confidence Calibration
    high_conf_n:          int    # responses claiming ≥90%
    high_conf_correct:    int    # of those, how many had correct verdicts
    calibration_rate:     Optional[float]
    calibration_pass:     bool   # within 10% of stated confidence

    # Per-query breakdown
    query_results:        list[QueryResult]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    mock: bool = False,
    retrieval_only: bool = False,
    n_chunks: int = 5,
    verbose: bool = True,
) -> BenchmarkResults:
    """
    Run the full 4-metric benchmark.

    Parameters
    ----------
    mock : bool
        Use mock LLM responses. Retrieval is still real.
    retrieval_only : bool
        Skip LLM calls entirely — run retrieval metrics only.
    n_chunks : int
        Chunks per collection for retrieval (default 5).
    verbose : bool
        Print per-query results as they run.
    """

    # ── Metric 1 + 2: Retrieval precision ────────────────────────────────────
    if verbose:
        print("=== METRIC 1 + 2: RETRIEVAL PRECISION ===\n")
    retrieval_metrics = measure_retrieval_precision(n=n_chunks, verbose=verbose)

    tsi_prec  = retrieval_metrics["tsi_precision"]
    nntr_prec = retrieval_metrics["nntr_precision"]
    xl_prec   = retrieval_metrics["cross_lingual_prec"]

    if retrieval_only:
        # Return partial results
        return BenchmarkResults(
            tsi_precision=tsi_prec,
            nntr_precision=nntr_prec,
            tsi_target_pass=tsi_prec >= 0.8 if tsi_prec else False,
            nntr_target_pass=nntr_prec >= 0.7 if nntr_prec else False,
            cross_lingual_prec=xl_prec,
            cross_lingual_pass=xl_prec >= 0.7 if xl_prec else False,
            concordance_n=0,
            concordance_correct=0,
            concordance_rate=None,
            concordance_pass=False,
            high_conf_n=0,
            high_conf_correct=0,
            calibration_rate=None,
            calibration_pass=False,
            query_results=[],
        )

    # ── Metrics 3 + 4: Reasoning quality ─────────────────────────────────────
    if verbose:
        print("\n=== METRIC 3: ANSWER CONCORDANCE (Green-tier) ===\n")

    query_results = []
    concordance_n = concordance_correct = 0
    high_conf_n = high_conf_correct = 0

    for query, exp_verdict, req_citations, exp_tsi, exp_nntr in REASONING_GROUND_TRUTH:
        if verbose:
            print(f"  Query: {query[:70]}...")

        response = reason(query, n_chunks=n_chunks, mock=mock)

        # Verdict match (case-insensitive, allow partial match)
        verdict_match = exp_verdict.upper() in response.verdict.upper()

        # Citation coverage
        found_cits = [c for c in req_citations
                      if any(c.lower() in a.lower() for a in response.citations)]
        missing_cits = [c for c in req_citations if c not in found_cits]

        is_green = response.confidence_tier == "GREEN"

        # Concordance: count Green-tier only
        concordance_eligible = is_green
        concordance_pass = None
        if concordance_eligible:
            concordance_n += 1
            if verdict_match:
                concordance_correct += 1
                concordance_pass = True
            else:
                concordance_pass = False

        # Calibration: count queries where model claims ≥90% confidence
        if response.confidence_pct >= 90:
            high_conf_n += 1
            if verdict_match:
                high_conf_correct += 1

        qr = QueryResult(
            query=query,
            expected_verdict=exp_verdict,
            actual_verdict=response.verdict,
            verdict_match=verdict_match,
            expected_citations=req_citations,
            actual_citations=response.citations,
            citations_found=found_cits,
            citations_missing=missing_cits,
            confidence_tier=response.confidence_tier,
            confidence_pct=response.confidence_pct,
            is_green_tier=is_green,
            concordance_eligible=concordance_eligible,
            concordance_pass=concordance_pass,
        )
        query_results.append(qr)

        if verbose:
            v_icon = "✓" if verdict_match else "✗"
            c_icon = "✓" if not missing_cits else "~"
            print(f"    {v_icon} Verdict: {response.verdict} (expected: {exp_verdict})")
            print(f"    {c_icon} Citations: found={found_cits}, missing={missing_cits}")
            print(f"    {response.confidence_tier} {response.confidence_pct}%")
            print()

    concordance_rate = concordance_correct / concordance_n if concordance_n else None
    calibration_rate = high_conf_correct / high_conf_n if high_conf_n else None

    results = BenchmarkResults(
        tsi_precision=tsi_prec,
        nntr_precision=nntr_prec,
        tsi_target_pass=tsi_prec >= 0.8 if tsi_prec else False,
        nntr_target_pass=nntr_prec >= 0.7 if nntr_prec else False,
        cross_lingual_prec=xl_prec,
        cross_lingual_pass=xl_prec >= 0.7 if xl_prec else False,
        concordance_n=concordance_n,
        concordance_correct=concordance_correct,
        concordance_rate=concordance_rate,
        concordance_pass=concordance_rate >= 0.75 if concordance_rate else False,
        high_conf_n=high_conf_n,
        high_conf_correct=high_conf_correct,
        calibration_rate=calibration_rate,
        calibration_pass=(
            calibration_rate >= 0.80 if calibration_rate else False
        ),
        query_results=query_results,
    )

    if verbose:
        _print_summary(results)

    return results


def _print_summary(r: BenchmarkResults) -> None:
    """Print the benchmark summary card."""
    def pct(v):
        return f"{v:.0%}" if v is not None else "N/A"
    def pass_fail(b):
        return "✓ PASS" if b else "✗ FAIL"

    print("\n" + "=" * 60)
    print("TRACKMIND BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  Metric 1 — TSI Retrieval Precision:     {pct(r.tsi_precision):<6}  target ≥80%  {pass_fail(r.tsi_target_pass)}")
    print(f"  Metric 1 — NNTR Retrieval Precision:    {pct(r.nntr_precision):<6}  target ≥70%  {pass_fail(r.nntr_target_pass)}")
    print(f"  Metric 2 — Cross-Lingual Precision:     {pct(r.cross_lingual_prec):<6}  target ≥70%  {pass_fail(r.cross_lingual_pass)}")
    print(f"  Metric 3 — Answer Concordance (Green):  {pct(r.concordance_rate):<6}  target ≥75%  {pass_fail(r.concordance_pass)}")
    print(f"             ({r.concordance_correct}/{r.concordance_n} Green-tier queries)")
    print(f"  Metric 4 — Confidence Calibration:      {pct(r.calibration_rate):<6}  target ≥80%  {pass_fail(r.calibration_pass)}")
    print(f"             ({r.high_conf_correct}/{r.high_conf_n} ≥90% conf queries correct)")
    print()

    passing = sum([
        r.tsi_target_pass, r.nntr_target_pass, r.cross_lingual_pass,
        r.concordance_pass, r.calibration_pass
    ])
    print(f"  Overall: {passing}/5 metrics passing targets")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackMind benchmark runner")
    parser.add_argument("--mock", "-m", action="store_true",
                        help="Use mock LLM responses (no API cost)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Run retrieval metrics only (no LLM calls)")
    parser.add_argument("--n", type=int, default=5,
                        help="Chunks per collection (default 5)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    results = run_benchmark(
        mock=args.mock,
        retrieval_only=args.retrieval_only,
        n_chunks=args.n,
        verbose=True,
    )

    if args.save:
        # Convert dataclasses to dict for JSON serialisation
        data = asdict(results)
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {args.save}")
