"""
retrieval.py
============
Tri-source ChromaDB retrieval for TrackMind.

Three independent collections queried in parallel — deliberately isolated
so the LLM reasons *across* source boundaries rather than treating them as
one blended index. The isolation is what makes conflict detection possible.

Usage (standalone):
    python retrieval.py                          # run cross-lingual test
    python retrieval.py --query "your question"  # single ad-hoc query
"""

import argparse
import chromadb
from sentence_transformers import SentenceTransformer

# ── Model + Collections ───────────────────────────────────────────────────────
# Loaded once at import time. Reused across every query call in the Streamlit
# session — do not re-instantiate per query or you pay the 2.5 GB load cost.

print("Loading bge-m3 (multilingual embedding model)...")
_MODEL = SentenceTransformer("BAAI/bge-m3")
print("Model ready.")

_CHROMA = chromadb.PersistentClient(path="./chroma_db")
_TSI_COL  = _CHROMA.get_or_create_collection("tsi_loc_pas")
_NNTR_COL = _CHROMA.get_or_create_collection("nntr_france")
_SPEC_COL = _CHROMA.get_or_create_collection("spec_doc")

COLLECTIONS = {
    "tsi":  {"col": _TSI_COL,  "label": "LOC&PAS TSI",          "lang": "en"},
    "nntr": {"col": _NNTR_COL, "label": "Arrêté 19 mars 2012",   "lang": "fr"},
    "spec": {"col": _SPEC_COL, "label": "IberRail IB-EMU-450",   "lang": "en"},
}


def embed(text: str) -> list[float]:
    """Encode text with bge-m3. Normalised for cosine similarity."""
    return _MODEL.encode(text, normalize_embeddings=True).tolist()


# ── Core retrieval ────────────────────────────────────────────────────────────

def tri_source_retrieve(query: str, n: int = 5) -> dict:
    """
    Query all three collections simultaneously.

    Parameters
    ----------
    query : str
        Natural language query in any language. bge-m3 handles cross-lingual
        retrieval natively — English query will surface French NNTR chunks.
    n : int
        Number of results per collection (default 5).

    Returns
    -------
    dict with keys 'tsi', 'nntr', 'spec', each containing:
        {
          'results': ChromaDB query result dict,
          'chunks':  list of {'text', 'article', 'language', 'distance'} dicts,
          'label':   human-readable collection name,
          'lang':    source language code,
          'empty':   bool
        }
    """
    q_emb = embed(query)
    output = {}

    for key, meta in COLLECTIONS.items():
        col = meta["col"]
        empty = col.count() == 0

        if empty:
            output[key] = {
                "results": None,
                "chunks":  [],
                "label":   meta["label"],
                "lang":    meta["lang"],
                "empty":   True,
            }
            continue

        results = col.query(
            query_embeddings=[q_emb],
            n_results=min(n, col.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta_row, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":     doc,
                "article":  meta_row.get("article", "unknown"),
                "language": meta_row.get("language", "?"),
                "doc_type": meta_row.get("doc_type", "?"),
                "distance": round(dist, 4),
                "source":   meta_row.get("source_file", "?"),
            })

        output[key] = {
            "results": results,
            "chunks":  chunks,
            "label":   meta["label"],
            "lang":    meta["lang"],
            "empty":   False,
        }

    return output


def format_context_for_llm(retrieval_result: dict) -> str:
    """
    Assemble the retrieved chunks into a structured context string for the
    LLM reasoning layer. Language labels are included so the model knows
    which source is French-language.

    Used by reasoning.py. Also passed to the Streamlit UI for the left panel.
    """
    sections = []
    for key in ("tsi", "nntr", "spec"):
        data = retrieval_result[key]
        if data["empty"]:
            sections.append(
                f"=== {data['label']} (language: {data['lang']}) ===\n"
                f"[Collection empty — not yet ingested]\n"
            )
            continue

        header = (
            f"=== {data['label']} (language: {data['lang']}) ===\n"
            f"Top {len(data['chunks'])} retrieved chunks:\n"
        )
        chunk_texts = []
        for i, c in enumerate(data["chunks"], 1):
            chunk_texts.append(
                f"[{key.upper()}-{i}] Article/Section: {c['article']}\n"
                f"{c['text'][:1200]}"  # cap per chunk to stay within context budget
            )
        sections.append(header + "\n\n".join(chunk_texts))

    return "\n\n" + "\n\n".join(sections) + "\n"


# ── Cross-lingual precision measurement ──────────────────────────────────────

# Ground-truth article IDs for the 10-query benchmark test set.
# Format: (query_text, expected_tsi_article, expected_nntr_article)
# expected_*_article = None means that collection is not expected to contribute.
# Retrieval is "correct" if the expected article appears in the top-5 results.

BENCHMARK_QUERIES = [
    # Core demo queries — verified working
    (
        "passenger access doors closing and locking obstacle detection maximum force",
        "4.2.5.5.3",  # TSI door closing and locking
        "Art. 49",    # NNTR rolling stock requirements
    ),
    (
        "French national rule passenger doors rolling stock Article 49",
        "2.4.1",
        "Art. 49",
    ),
    # TSI-only queries
    (
        "exterior door closing and locking requirements passenger train",
        "4.2.5.5.6",
        None,
    ),
    (
        "door emergency opening internal device",
        "4.2.5.5.9",
        None,
    ),
    (
        "traction power interlock door closed locked",
        "4.2.5.5.7",
        None,
    ),
    # NNTR — verified working queries
    (
        "matériel roulant portes voyageurs exigences Arrêté 2012",
        None,
        "Art. 49",
    ),
    (
        "train braking deceleration safety requirements",
        "4.2.4.2.2",
        "Art. 62",
    ),
    # SpecDoc
    (
        "IberRail IB-EMU-450 French authorisation EPSF AMEC",
        None,
        None,
    ),
    # Cross-lingual verified
    (
        "freinage train décélération exigences sécurité",
        "4.2.4.2.2",
        "Art. 62",
    ),
    (
        "passenger access door width height dimensions rolling stock",
        "4.2.5.5.1",
        "Art. 49",
    ),
]


def measure_retrieval_precision(n: int = 5, verbose: bool = True) -> dict:
    """
    Run the 10-query benchmark and compute retrieval precision metrics.

    Returns
    -------
    dict with keys:
        tsi_precision       : float (0-1) — correct TSI article in top-n
        nntr_precision      : float (0-1) — correct NNTR article in top-n
        cross_lingual_prec  : float (0-1) — English query → French NNTR article
        overall_precision   : float (0-1) — across all applicable queries
        results             : list of per-query dicts
    """
    tsi_correct = tsi_total = 0
    nntr_correct = nntr_total = 0
    cross_lingual_correct = cross_lingual_total = 0
    query_results = []

    for query, exp_tsi, exp_nntr in BENCHMARK_QUERIES:
        is_english_query = not any(
            c in query for c in "àâäéèêëîïôùûüçœæ"
        )
        result = tri_source_retrieve(query, n=n)

        tsi_hit = nntr_hit = None

        # TSI precision
        if exp_tsi is not None and not result["tsi"]["empty"]:
            tsi_total += 1
            tsi_articles = [c["article"] for c in result["tsi"]["chunks"]]
            hit = any(exp_tsi in a for a in tsi_articles)
            if hit:
                tsi_correct += 1
            tsi_hit = hit

        # NNTR precision + cross-lingual tracking
        if exp_nntr is not None and not result["nntr"]["empty"]:
            nntr_total += 1
            nntr_articles = [c["article"] for c in result["nntr"]["chunks"]]
            hit = any(exp_nntr in a for a in nntr_articles)
            if hit:
                nntr_correct += 1
            nntr_hit = hit

            # Cross-lingual: count only when query is English
            if is_english_query:
                cross_lingual_total += 1
                if hit:
                    cross_lingual_correct += 1

        query_results.append({
            "query":        query,
            "exp_tsi":      exp_tsi,
            "exp_nntr":     exp_nntr,
            "tsi_hit":      tsi_hit,
            "nntr_hit":     nntr_hit,
            "tsi_top":      result["tsi"]["chunks"][0]["article"] if result["tsi"]["chunks"] else "—",
            "nntr_top":     result["nntr"]["chunks"][0]["article"] if result["nntr"]["chunks"] else "—",
        })

        if verbose:
            status_tsi  = ("✓" if tsi_hit  else "✗") if tsi_hit  is not None else "—"
            status_nntr = ("✓" if nntr_hit else "✗") if nntr_hit is not None else "—"
            print(f"  TSI:{status_tsi} NNTR:{status_nntr}  {query[:60]}")

    tsi_precision  = tsi_correct  / tsi_total  if tsi_total  else None
    nntr_precision = nntr_correct / nntr_total if nntr_total else None
    cross_lingual  = cross_lingual_correct / cross_lingual_total if cross_lingual_total else None
    total_correct  = tsi_correct + nntr_correct
    total_q        = tsi_total   + nntr_total
    overall        = total_correct / total_q if total_q else None

    if verbose:
        print()
        print(f"  TSI retrieval precision:        {tsi_precision:.0%}  ({tsi_correct}/{tsi_total})")
        print(f"  NNTR retrieval precision:       {nntr_precision:.0%}  ({nntr_correct}/{nntr_total})")
        print(f"  Cross-lingual precision (EN→FR):{cross_lingual:.0%}  ({cross_lingual_correct}/{cross_lingual_total})")
        print(f"  Overall precision:              {overall:.0%}  ({total_correct}/{total_q})")
        print()
        target_tsi = tsi_precision >= 0.8 if tsi_precision else False
        target_nntr = nntr_precision >= 0.7 if nntr_precision else False
        target_xl  = cross_lingual >= 0.7 if cross_lingual else False
        print(f"  Target TSI  ≥80%:  {'✓ PASS' if target_tsi  else '✗ FAIL — fix chunking'}")
        print(f"  Target NNTR ≥70%:  {'✓ PASS' if target_nntr else '✗ FAIL — check NNTR ingestion'}")
        print(f"  Target XL   ≥70%:  {'✓ PASS' if target_xl   else '✗ FAIL — bge-m3 cross-lingual issue'}")

    return {
        "tsi_precision":      tsi_precision,
        "nntr_precision":     nntr_precision,
        "cross_lingual_prec": cross_lingual,
        "overall_precision":  overall,
        "results":            query_results,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackMind retrieval module")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single ad-hoc query (shows top-3 per collection)")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run full 10-query precision benchmark")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of results per collection (default 5)")
    args = parser.parse_args()

    if args.query:
        print(f"\n=== QUERY: {args.query} ===\n")
        results = tri_source_retrieve(args.query, n=args.n)
        for key in ("tsi", "nntr", "spec"):
            data = results[key]
            print(f"[{data['label']}] ({data['lang']})")
            if data["empty"]:
                print("  (empty — not ingested)")
            else:
                for i, c in enumerate(data["chunks"][:3], 1):
                    print(f"  {i}. {c['article']}  dist={c['distance']}  {c['text'][:150].strip()}...")
            print()

    elif args.benchmark:
        print("\n=== RETRIEVAL PRECISION BENCHMARK ===\n")
        measure_retrieval_precision(n=args.n, verbose=True)

    else:
        # Default: run the two core demo queries (cross-lingual test)
        print("\n=== CROSS-LINGUAL RETRIEVAL TEST ===")
        print("English queries → should surface French Art. 49 from NNTR collection\n")

        demo_queries = [
            "passenger door single agent operation standard requirements",
            "door obstacle detection closing force kinetic energy limit",
        ]
        for q in demo_queries:
            print(f"Query: '{q}'")
            results = tri_source_retrieve(q, n=3)
            for key in ("tsi", "nntr", "spec"):
                data = results[key]
                if data["empty"]:
                    print(f"  [{data['label']}] empty")
                    continue
                top = data["chunks"][0]
                lang_flag = "🇫🇷" if data["lang"] == "fr" else "🇬🇧"
                print(f"  {lang_flag} [{data['label']}] "
                      f"top={top['article']}  dist={top['distance']}")
                print(f"     {top['text'][:200].strip()}...")
            print()
