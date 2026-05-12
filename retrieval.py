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
  retrieve_regulatory()          — queries TSI + NNTR only (no spec)
  retrieve_with_session_spec()   — queries TSI + NNTR from ChromaDB, uses in-memory
                                   spec chunks passed from the upload session
  tri_source_retrieve()          — compatibility wrapper that returns TSI + NNTR
                                   plus an empty SPEC slot

Usage (standalone):
    python retrieval.py                          # run cross-lingual test
    python retrieval.py --query "your question"  # single ad-hoc query
"""

import argparse
import re
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


def _empty_source(meta: dict) -> dict:
    return {
        "results": None,
        "chunks":  [],
        "label":   meta["label"],
        "lang":    meta["lang"],
        "empty":   True,
    }


def empty_spec_source() -> dict:
    return {
        "results": None,
        "chunks":  [],
        "label":   "Spec (no file uploaded)",
        "lang":    "en",
        "empty":   True,
    }


def _chunks_from_query_results(results: dict) -> list[dict]:
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
    return chunks


def _query_regulatory_source(source_key: str, query: str, n: int) -> dict:
    if source_key not in COLLECTIONS:
        raise ValueError(f"Unknown regulatory source: {source_key}")

    meta = COLLECTIONS[source_key]
    col = meta["col"]
    if col.count() == 0:
        return _empty_source(meta)

    results = col.query(
        query_embeddings=[embed(query)],
        n_results=min(n, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return {
        "results": results,
        "chunks":  _chunks_from_query_results(results),
        "label":   meta["label"],
        "lang":    meta["lang"],
        "empty":   False,
    }


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve_regulatory(query: str, n: int = 5) -> dict:
    """
    Query TSI and NNTR collections only (no spec).

    Returns dict with keys 'tsi', 'nntr', each containing:
        { results, chunks, label, lang, empty }
    """
    return {
        key: _query_regulatory_source(key, query, n)
        for key in COLLECTIONS
    }


def retrieve_regulatory_article(source_key: str, article: str, n: int = 3) -> list[dict]:
    """Fetch regulatory chunks by exact article/section metadata."""
    if source_key not in COLLECTIONS:
        raise ValueError(f"Unknown regulatory source: {source_key}")

    col = COLLECTIONS[source_key]["col"]
    if col.count() == 0:
        return []

    results = col.get(
        where={"article": article},
        limit=n,
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta_row in zip(results.get("documents", []), results.get("metadatas", [])):
        chunks.append({
            "text":     doc,
            "article":  meta_row.get("article", "unknown"),
            "language": meta_row.get("language", "?"),
            "doc_type": meta_row.get("doc_type", "?"),
            "distance": None,
            "source":   meta_row.get("source_file", "?"),
        })
    return chunks


def _chunk_key(chunk: dict) -> tuple[str, str]:
    return (str(chunk.get("article", "")).lower(), str(chunk.get("text", ""))[:160].lower())


def _append_unique(chunks: list[dict], chunk: dict) -> None:
    key = _chunk_key(chunk)
    if all(_chunk_key(existing) != key for existing in chunks):
        chunks.append(chunk)


def _extract_source_refs(text: str, source_key: str) -> list[str]:
    refs = []
    if source_key == "tsi":
        refs.extend(re.findall(r"(?:TSI|LOC&PAS|LOC PAS).{0,80}?(?:art\.?|article|section|sec\.?|clause|cl\.?)\s*(\d+(?:\.\d+)+)", text, flags=re.IGNORECASE | re.DOTALL))
        refs.extend(re.findall(r"\b(Article\s+\d+)\b", text, flags=re.IGNORECASE))
    elif source_key == "nntr":
        refs.extend(f"Art. {num}" for num in re.findall(r"(?:Arrêté|Arrete|NNTR|RFN|French).{0,80}?(?:art\.?|article)\s*(\d+(?:er|ère|re|nd)?)", text, flags=re.IGNORECASE | re.DOTALL))

    unique = []
    for ref in refs:
        ref = re.sub(r"\s+", " ", ref).strip().rstrip(".")
        if ref and ref not in unique:
            unique.append(ref)
    return unique


def _boost_regulatory_chunks(source_key: str, combined_text: str, max_chunks: int) -> list[dict]:
    text = combined_text.lower()
    articles = []

    if source_key == "tsi":
        door_signal = any(token in text for token in ("door", "doors", "obstacle", "14752", "passenger access"))
        if door_signal:
            articles.extend(["4.2.5.5.3", "4.2.5.5.1", "4.2.5.5.2", "4.2.5.5.5", "4.2.5.5.6"])
    elif source_key == "nntr":
        french_signal = any(token in text for token in ("arrêté", "arrete", "french", "rfn", "nf f31", "amec", "epsf", "door", "obstacle"))
        if french_signal:
            articles.append("Art. 49")

    articles.extend(_extract_source_refs(combined_text, source_key))

    boosted = []
    for article in articles:
        for chunk in retrieve_regulatory_article(source_key, article, n=2):
            _append_unique(boosted, chunk)
            if len(boosted) >= max_chunks:
                return boosted
    return boosted


def _spec_keyword_bonus(query: str, chunk: dict) -> float:
    """Small deterministic boost so compliance-conflict questions surface the right SPEC evidence."""
    q = (query or "").lower()
    article = str(chunk.get("metadata", {}).get("article", "")).lower()
    text = str(chunk.get("text", "")).lower()
    combined = f"{article} {text}"

    asks_conflict = any(token in q for token in (
        "conflict",
        "gap",
        "non-compliance",
        "non compliance",
        "não conforme",
        "conflito",
    ))
    asks_three_sources = (
        any(token in q for token in ("tsi", "loc&pas", "loc pas"))
        and any(token in q for token in ("french", "rfn", "nntr", "arrêté", "arrete"))
        and any(token in q for token in ("spec", "specification", "uploaded"))
    )
    asks_nf_f31 = any(token in q for token in ("nf f31", "f31-054", "section 6.3", "obstacle"))

    bonus = 0.0
    if asks_conflict or asks_three_sources:
        if article.startswith("4.2.3") or "conflict analysis" in combined:
            bonus += 0.45
        if "compliance position" in combined or "resolution plan" in combined:
            bonus += 0.35
        if "conflict" in combined:
            bonus += 0.30
        if "open issue" in combined or re.search(r"\boi-\d+\b", combined):
            bonus += 0.25
        if article in {"1.2", "4.0", "4.1"}:
            bonus -= 0.18

    if asks_nf_f31:
        if "nf f31-054" in combined or "f31-054" in combined:
            bonus += 0.25
        if "en 14752" in combined or "14752" in combined:
            bonus += 0.10
        if "not yet conducted" in combined or "not been conducted" in combined:
            bonus += 0.25
        if "not completed" in combined or "incomplete" in combined:
            bonus += 0.20

    return bonus


def rank_session_spec_chunks(query: str, session_spec_chunks: list[dict], n: int = 3) -> list[dict]:
    """Rank uploaded spec chunks against arbitrary text without writing to ChromaDB."""
    import numpy as np

    if not session_spec_chunks:
        return []

    q_vec = np.array(embed(query))
    scored = []
    for chunk in session_spec_chunks:
        if "_embedding" not in chunk:
            chunk["_embedding"] = embed(chunk["text"])
        c_vec = np.array(chunk["_embedding"])
        distance = round(1.0 - float(np.dot(q_vec, c_vec)), 4)
        adjusted = round(distance - _spec_keyword_bonus(query, chunk), 4)
        scored.append((adjusted, distance, chunk))

    scored.sort(key=lambda x: x[0])
    chunks = []
    for _, dist, chunk in scored[:n]:
        chunks.append({
            "text":     chunk["text"],
            "article":  chunk["metadata"].get("article", "unknown"),
            "language": chunk["metadata"].get("language", "en"),
            "doc_type": chunk["metadata"].get("doc_type", "SPEC"),
            "distance": dist,
            "source":   chunk["metadata"].get("source_file", "uploaded"),
        })
    return chunks


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
    # Rank spec chunks by cosine similarity in memory
    spec_result = empty_spec_source()

    if session_spec_chunks:
        spec_chunks = rank_session_spec_chunks(query, session_spec_chunks, n=n)

        spec_result = {
            "results": None,
            "chunks":  spec_chunks,
            "label":   "Uploaded Spec",
            "lang":    "en",
            "empty":   False,
        }

    combined_text = " ".join([query] + [c["text"] for c in spec_result["chunks"][:4]])
    semantic_regulatory = retrieve_regulatory(query, n=n)
    regulatory = {}
    for key in ("tsi", "nntr"):
        chunks = []
        for chunk in _boost_regulatory_chunks(key, combined_text, max_chunks=n):
            _append_unique(chunks, chunk)
        for chunk in semantic_regulatory[key]["chunks"]:
            _append_unique(chunks, chunk)
            if len(chunks) >= n:
                break
        regulatory[key] = {**semantic_regulatory[key], "chunks": chunks, "empty": semantic_regulatory[key]["empty"] and not chunks}

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
        "spec": empty_spec_source(),
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
