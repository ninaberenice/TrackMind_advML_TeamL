"""
retrieval.py
============
Tri-source ChromaDB retrieval for TrackMind.

Architecture (updated):
  - TSI (tsi_loc_pas) and NNTR (nntr_france) are permanently ingested in ChromaDB.
  - SPEC is NOT stored in ChromaDB. Instead, the user uploads a spec PDF per session.
    The spec is chunked in memory and injected directly into the LLM context.
    This means manufacturer specs never persist in the vector DB — cleaner, safer,
    and architecturally correct (the spec is the "subject under test", not a rule source).

Key functions:
  tri_source_retrieve()          — original, queries all 3 ChromaDB collections
                                   (spec_doc collection will be empty/ignored now)
  retrieve_regulatory()          — queries TSI + NNTR only (no spec)
  retrieve_with_session_spec()   — queries TSI + NNTR from ChromaDB, uses in-memory
                                   spec chunks passed from the upload session

Usage (standalone):
    python retrieval.py                          # run cross-lingual test
    python retrieval.py --query "your question"  # single ad-hoc query
"""

import argparse
import chromadb
from sentence_transformers import SentenceTransformer

# ── Model + Collections ───────────────────────────────────────────────────────

print("Loading bge-m3 (multilingual embedding model)...")
_MODEL = SentenceTransformer("BAAI/bge-m3")
print("Model ready.")

_CHROMA = chromadb.PersistentClient(path="./chroma_db")
_TSI_COL  = _CHROMA.get_or_create_collection("tsi_loc_pas")
_NNTR_COL = _CHROMA.get_or_create_collection("nntr_france")
# spec_doc collection is no longer used — specs are session-only in memory

COLLECTIONS = {
    "tsi":  {"col": _TSI_COL,  "label": "LOC&PAS TSI",          "lang": "en"},
    "nntr": {"col": _NNTR_COL, "label": "Arrêté 19 mars 2012",   "lang": "fr"},
}


def embed(text: str) -> list[float]:
    """Encode text with bge-m3. Normalised for cosine similarity."""
    return _MODEL.encode(text, normalize_embeddings=True).tolist()


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve_regulatory(query: str, n: int = 5) -> dict:
    """
    Query TSI and NNTR collections only (no spec).

    Returns dict with keys 'tsi', 'nntr', each containing:
        { results, chunks, label, lang, empty }
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


def retrieve_with_session_spec(
    query: str,
    session_spec_chunks: list[dict],
    n: int = 5,
) -> dict:
    """
    Query TSI + NNTR from ChromaDB, and rank the uploaded session spec chunks
    by cosine similarity in Python (no ChromaDB write).

    Parameters
    ----------
    query : str
        Compliance question.
    session_spec_chunks : list[dict]
        In-memory spec chunks from the uploaded PDF. Each chunk is:
        { 'id', 'text', 'metadata': { 'article', 'doc_type', ... } }
        These come from trackmind_chunker.chunk_generic() called at upload time.
    n : int
        Number of results per source.

    Returns
    -------
    dict with keys 'tsi', 'nntr', 'spec' — same shape as tri_source_retrieve().
    """
    import numpy as np

    # Get regulatory results from ChromaDB
    regulatory = retrieve_regulatory(query, n=n)

    # Rank spec chunks by cosine similarity in memory
    spec_result = {
        "results": None,
        "chunks":  [],
        "label":   "Uploaded Spec",
        "lang":    "en",
        "empty":   True,
    }

    if session_spec_chunks:
        q_emb = embed(query)
        q_vec = np.array(q_emb)

        scored = []
        for chunk in session_spec_chunks:
            if "_embedding" not in chunk:
                # Embed on first use and cache on the chunk object
                chunk["_embedding"] = embed(chunk["text"])
            c_vec = np.array(chunk["_embedding"])
            # Cosine similarity (both normalised → dot product)
            sim = float(np.dot(q_vec, c_vec))
            distance = round(1.0 - sim, 4)  # convert to distance for consistency
            scored.append((distance, chunk))

        scored.sort(key=lambda x: x[0])
        top_n = scored[:n]

        spec_chunks = []
        for dist, chunk in top_n:
            spec_chunks.append({
                "text":     chunk["text"],
                "article":  chunk["metadata"].get("article", "unknown"),
                "language": chunk["metadata"].get("language", "en"),
                "doc_type": chunk["metadata"].get("doc_type", "SPEC"),
                "distance": dist,
                "source":   chunk["metadata"].get("source_file", "uploaded"),
            })

        spec_result = {
            "results": None,
            "chunks":  spec_chunks,
            "label":   "Uploaded Spec",
            "lang":    "en",
            "empty":   False,
        }

    return {
        "tsi":  regulatory["tsi"],
        "nntr": regulatory["nntr"],
        "spec": spec_result,
    }


# ── Legacy tri_source_retrieve (kept for backward compat / benchmark) ─────────

def tri_source_retrieve(query: str, n: int = 5) -> dict:
    """
    Original function — now returns TSI + NNTR from ChromaDB and an empty spec
    slot (since spec is no longer stored in ChromaDB). Kept for benchmark.py
    and any callers that expect all three keys.
    """
    regulatory = retrieve_regulatory(query, n=n)
    return {
        "tsi":  regulatory["tsi"],
        "nntr": regulatory["nntr"],
        "spec": {
            "results": None,
            "chunks":  [],
            "label":   "Spec (no file uploaded)",
            "lang":    "en",
            "empty":   True,
        },
    }


def format_context_for_llm(retrieval_result: dict) -> str:
    """
    Assemble retrieved chunks into a structured context string for the LLM.
    Works with both tri_source_retrieve() and retrieve_with_session_spec().
    """
    sections = []
    for key in ("tsi", "nntr", "spec"):
        data = retrieval_result.get(key)
        if data is None:
            continue
        if data["empty"]:
            sections.append(
                f"=== {data['label']} (language: {data['lang']}) ===\n"
                f"[No document available for this source]\n"
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
                f"{c['text'][:1200]}"
            )
        sections.append(header + "\n\n".join(chunk_texts))

    return "\n\n" + "\n\n".join(sections) + "\n"


# ── Cross-lingual precision measurement ──────────────────────────────────────

BENCHMARK_QUERIES = [
    (
        "passenger access doors closing and locking obstacle detection maximum force",
        "4.2.5.5.3",
        "Art. 49",
    ),
    (
        "French national rule passenger doors rolling stock Article 49",
        "2.4.1",
        "Art. 49",
    ),
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
    (
        "passenger access door width height dimensions rolling stock",
        "4.2.5.5.1",
        "Art. 49",
    ),
    (
        "freinage train décélération exigences sécurité",
        "4.2.4.2.2",
        "Art. 62",
    ),
    (
        "door obstacle detection closing force kinetic energy limit",
        "4.2.5.5.3",
        "Art. 49",
    ),
]


def measure_retrieval_precision(n: int = 5, verbose: bool = True) -> dict:
    """
    Run the benchmark and compute retrieval precision metrics (TSI + NNTR only).
    Spec is excluded since it's now session-uploaded, not pre-ingested.
    """
    tsi_correct = tsi_total = 0
    nntr_correct = nntr_total = 0
    cross_lingual_correct = cross_lingual_total = 0
    query_results = []

    for query, exp_tsi, exp_nntr in BENCHMARK_QUERIES:
        is_english_query = not any(c in query for c in "àâäéèêëîïôùûüçœæ")
        result = retrieve_regulatory(query, n=n)

        tsi_hit = nntr_hit = None

        if exp_tsi is not None and not result["tsi"]["empty"]:
            tsi_total += 1
            tsi_articles = [c["article"] for c in result["tsi"]["chunks"]]
            hit = any(exp_tsi in a for a in tsi_articles)
            if hit:
                tsi_correct += 1
            tsi_hit = hit

        if exp_nntr is not None and not result["nntr"]["empty"]:
            nntr_total += 1
            nntr_articles = [c["article"] for c in result["nntr"]["chunks"]]
            hit = any(exp_nntr in a for a in nntr_articles)
            if hit:
                nntr_correct += 1
            nntr_hit = hit

            if is_english_query:
                cross_lingual_total += 1
                if hit:
                    cross_lingual_correct += 1

        query_results.append({
            "query":    query,
            "exp_tsi":  exp_tsi,
            "exp_nntr": exp_nntr,
            "tsi_hit":  tsi_hit,
            "nntr_hit": nntr_hit,
            "tsi_top":  result["tsi"]["chunks"][0]["article"] if result["tsi"]["chunks"] else "—",
            "nntr_top": result["nntr"]["chunks"][0]["article"] if result["nntr"]["chunks"] else "—",
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
        print(f"  TSI retrieval precision:         {tsi_precision:.0%}  ({tsi_correct}/{tsi_total})")
        print(f"  NNTR retrieval precision:        {nntr_precision:.0%}  ({nntr_correct}/{nntr_total})")
        print(f"  Cross-lingual precision (EN→FR): {cross_lingual:.0%}  ({cross_lingual_correct}/{cross_lingual_total})")
        print(f"  Overall precision:               {overall:.0%}  ({total_correct}/{total_q})")
        target_tsi  = tsi_precision  >= 0.8 if tsi_precision  else False
        target_nntr = nntr_precision >= 0.7 if nntr_precision else False
        target_xl   = cross_lingual  >= 0.7 if cross_lingual  else False
        print(f"  Target TSI  ≥80%: {'✓ PASS' if target_tsi  else '✗ FAIL'}")
        print(f"  Target NNTR ≥70%: {'✓ PASS' if target_nntr else '✗ FAIL'}")
        print(f"  Target XL   ≥70%: {'✓ PASS' if target_xl   else '✗ FAIL'}")

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
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--benchmark", "-b", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        print(f"\n=== QUERY: {args.query} ===\n")
        results = retrieve_regulatory(args.query, n=args.n)
        for key in ("tsi", "nntr"):
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
        print("\n=== CROSS-LINGUAL RETRIEVAL TEST ===")
        demo_queries = [
            "passenger door single agent operation standard requirements",
            "door obstacle detection closing force kinetic energy limit",
        ]
        for q in demo_queries:
            print(f"Query: '{q}'")
            results = retrieve_regulatory(q, n=3)
            for key in ("tsi", "nntr"):
                data = results[key]
                if data["empty"]:
                    print(f"  [{data['label']}] empty")
                    continue
                top = data["chunks"][0]
                lang_flag = "🇫🇷" if data["lang"] == "fr" else "🇬🇧"
                print(f"  {lang_flag} [{data['label']}] top={top['article']}  dist={top['distance']}")
                print(f"     {top['text'][:200].strip()}...")
            print()
