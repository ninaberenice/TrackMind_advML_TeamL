"""
trackmind_chunker.py
====================
Improved document chunkers for the TrackMind ingestion pipeline.

Replaces the single `extract_chunks()` function in the notebook with
document-type-aware implementations that fix the three root bugs
diagnosed below.

ROOT CAUSE ANALYSIS
-------------------

Bug 1 — NNTR: Wrong pattern splits on page headers, not articles
  The original regex `r'(?=\n(?:Article\s+\d+|Art\.\s*\d+|\d+[\d\.]*)\s+[A-ZÀ-Ÿa-z])'`
  matches "28 mars 2012" (the repeated page header) because "28" matches
  `\d+[\d\.]*` and "mars" matches `[A-ZÀ-Ÿa-z]`. The 46-page PDF has 46
  page headers, each triggering a false split. This is why the notebook
  reports 71 chunks but they mostly start with "section_N" (page header
  fragments) rather than "Art. N".

Bug 2 — NNTR: Em-dash separator blocks the real split
  Every article in the Arrêté du 19 mars 2012 uses the format:
      "Art. 49. −Sans préjudice..."
  where "−" is Unicode U+2212 MINUS SIGN (not ASCII hyphen-minus U+002D).
  The original pattern requires `\s+[A-ZÀ-Ÿa-z]` immediately after the
  article number match, but what actually follows is ". −S" — a dot, a
  space, an em-dash, then text. The lookahead never fires for real articles.
  This is why Art. 49 lands in a big blob chunk (chunk 18) starting with
  "28 mars 2012" rather than its own chunk.

Bug 3 — NNTR: article_id extraction falls back to "section_N"
  Because the split doesn't land on the actual "Art. 49. −" line, the
  metadata field `article` is set to "section_18" for the chunk containing
  Art. 49. Vector search returns this chunk but the query router finds
  "section_18", not "Art. 49", so the retrieval *looks* wrong even when
  the content is approximately correct.

TSI (loc_pas_tsi.pdf) STATUS
-----------------------------
The TSI chunking is *structurally working* — the same pattern correctly
splits on "4.2.5.5.1.\nGeneral" style headers (numeric sections start a
line and are followed by a title word that begins with a capital letter).
The TSI produces 1277 chunks with ~1080 having clean section-number IDs.

Two TSI issues worth fixing for retrieval quality:
  a) The EUR-Lex consolidated version has amendment markers (▼M5, ►B, ►M3)
     that appear as noise tokens inside chunks. Strip them for cleaner text.
  b) The table-of-contents pages at the front produce ~197 garbage chunks
     (chunks where the first line is a page-version stamp like
     "02014R1302 — EN — 27.04.2025 — 006.001 — 42"). These don't hurt
     retrieval badly but inflate chunk count and waste embedding budget.
     The improved TSI chunker strips them.
"""

import re
import fitz  # PyMuPDF


# ── NNTR: Arrêté du 19 mars 2012 ─────────────────────────────────────────────

# Repeated page header in the Journal Officiel PDF.
# Every one of the 46 pages starts with exactly this block.
_NNTR_HEADER_RE = re.compile(
    r'28 mars 2012\s*\n'
    r'JOURNAL OFFICIEL DE LA R[EÉ]PUBLIQUE FRAN[CÇ]AISE\s*\n'
    r'Texte 36 sur 102\s*\n'
    r'[.\s]*\n',
    re.IGNORECASE,
)

# The em-dash U+2212 (−) used as article body separator in the Arrêté.
# Also accept ASCII hyphen-minus (−) as defensive fallback.
_NNTR_ART_SPLIT_RE = re.compile(
    r'(?=\nArt\.\s+\d+(?:er|ère|re|nd|ème)?[\w]*\.?\s*[.−\-])'
)

# Structural dividers above article level (TITRE, CHAPITRE, SECTION, ANNEXE).
_NNTR_DIVIDERS_RE = re.compile(
    r'(?=\n(?:TITRE|CHAPITRE|Section|ANNEXE)\s+[IVXivx\d]+)'
)

# Pattern to extract article number from the start of a chunk.
_NNTR_ART_NUM_RE = re.compile(
    r'^Art\.\s+(\d+(?:er|ère|re|nd)?)',
    re.IGNORECASE,
)

_NNTR_DIVIDER_ID_RE = re.compile(
    r'^(TITRE\s+[IVXivx\d]+|CHAPITRE\s+[IVXivx\d]+|Section\s+\d+|ANNEXE\s+[IVXivx\d]+)',
    re.IGNORECASE,
)


def _strip_nntr_headers(text: str) -> str:
    """Remove repeating Journal Officiel page headers from extracted text."""
    text = _NNTR_HEADER_RE.sub('\n', text)
    # Clean up any remaining isolated page-decoration dot lines
    text = re.sub(r'\n\s*\.\s*\n\s*\.\s*\n', '\n', text)
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def chunk_french_nntr(pdf_path: str, doc_type: str = 'NNTR_FRANCE') -> list[dict]:
    """
    Article-level chunker for the Arrêté du 19 mars 2012 (French NNTR).

    Produces one chunk per article, plus preamble chunks for TITRE/CHAPITRE
    structural markers. Each chunk carries metadata including the clean
    article identifier (e.g. "Art. 49"), document type, language, and
    source file path.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to france_nntr.pdf.
    doc_type : str
        Collection identifier stored in chunk metadata.

    Returns
    -------
    list[dict]
        List of chunk dicts with keys: id, text, metadata.
        Ready to pass directly to ingest().
    """
    print(f'  Reading {pdf_path}...')
    doc = fitz.open(pdf_path)
    raw_text = ''.join(page.get_text() for page in doc)

    if not raw_text.strip():
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    # Step 1: Strip page headers
    clean = _strip_nntr_headers(raw_text)

    # Step 2: Split on Art. XX. pattern
    # We combine structural dividers and article splits so that TITRE/CHAPITRE
    # headings become their own preamble chunks providing context for embedding.
    combined_pattern = re.compile(
        r'(?=\nArt\.\s+\d+(?:er|ère|re|nd|ème)?[\w]*\.?\s*[.−\-])'
        r'|(?=\n(?:TITRE|CHAPITRE|Section\s+\d|ANNEXE)\s+[IVXivx\d])'
    )
    raw_chunks = combined_pattern.split(clean)

    chunks = []
    for i, piece in enumerate(raw_chunks):
        piece = piece.strip()

        # Identify type BEFORE the length filter so short structural headers
        # (e.g. "TITRE Ier\nDISPOSITIONS GÉNÉRALES" = 32 chars) are not dropped.
        art_match = _NNTR_ART_NUM_RE.match(piece)
        div_match = _NNTR_DIVIDER_ID_RE.match(piece)

        # Drop fragments that are neither article nor structural, and too short
        if not art_match and not div_match and len(piece) < 50:
            continue
        if len(piece) < 10:
            continue

        if art_match:
            article_id = f'Art. {art_match.group(1)}'
        elif div_match:
            article_id = div_match.group(1).strip()
        else:
            article_id = 'preamble'

        header = f"[Arrêté du 19 mars 2012, {article_id}] "
        augmented_text = header + piece

        chunks.append({
            'id': f'{doc_type}_{i}',
            'text': augmented_text,
            'metadata': {
                'article': article_id,
                'doc_type': doc_type,
                'subsystem': 'general',
                'chunk_index': i,
                'source_file': pdf_path,
                'language': 'fr',
            },
        })

    print(f'  Extracted {len(chunks)} chunks from NNTR.')
    return chunks


# ── TSI: LOC&PAS Commission Regulation ───────────────────────────────────────

# EUR-Lex consolidated document version stamp that appears at page breaks.
# e.g. "02014R1302 — EN — 27.04.2025 — 006.001 — 42"
_TSI_VERSION_STAMP_RE = re.compile(
    r'02014R1302\s*—\s*EN\s*—\s*[\d.]+\s*—\s*[\d.]+\s*—\s*\d+\s*\n?'
)

# EUR-Lex amendment markers injected inline: ▼M5, ►B, ►M3, ◄
_TSI_AMEND_MARKERS_RE = re.compile(r'[▼►◄]\s*[MB]\d*\s*|►C\d*\s*')

# TSI section header pattern: "4.2.5.5.3.\nDoor closing and locking"
# or "4.2.5.5.3 \nDoor closing" (space variant)
_TSI_SECTION_SPLIT_RE = re.compile(
    r'(?=\n\d+\.\d+[\.\d]*\.?\s+[A-Z])'
)

# Article-level split for the binding articles (Article 1, Article 2…)
_TSI_ARTICLE_SPLIT_RE = re.compile(
    r'(?=\nArticle\s+\d+\b)'
)

_TSI_SECTION_ID_RE = re.compile(r'^(\d+\.\d+[\.\d]*)\.?\s')
_TSI_ARTICLE_ID_RE = re.compile(r'^(Article\s+\d+)', re.IGNORECASE)


def _strip_tsi_noise(text: str) -> str:
    """Remove EUR-Lex version stamps and amendment markers from TSI text."""
    text = _TSI_VERSION_STAMP_RE.sub('', text)
    text = _TSI_AMEND_MARKERS_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def chunk_tsi_loc_pas(pdf_path: str, doc_type: str = 'TSI_LOC_PAS') -> list[dict]:
    """
    Section-level chunker for the LOC&PAS TSI (EU 1302/2014 consolidated).

    Splits on numeric section headers (4.2.5.5.3.) and top-level Articles,
    producing one chunk per section. Strips EUR-Lex amendment markers and
    version stamps that otherwise pollute chunk text.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to loc_pas_tsi.pdf.
    doc_type : str
        Collection identifier stored in chunk metadata.

    Returns
    -------
    list[dict]
        List of chunk dicts ready for ingest().
    """
    print(f'  Reading {pdf_path}...')
    doc = fitz.open(pdf_path)
    raw_text = ''.join(page.get_text() for page in doc)

    if not raw_text.strip():
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    clean = _strip_tsi_noise(raw_text)

    combined_pattern = re.compile(
        r'(?=\n\d+\.\d+[\.\d]*\.?\s+[A-Z])'
        r'|(?=\nArticle\s+\d+\b)'
    )
    raw_chunks = combined_pattern.split(clean)

    chunks = []
    for i, piece in enumerate(raw_chunks):
        piece = piece.strip()
        if len(piece) < 80:
            continue

        sec_match = _TSI_SECTION_ID_RE.match(piece)
        art_match = _TSI_ARTICLE_ID_RE.match(piece)

        if sec_match:
            article_id = sec_match.group(1)
        elif art_match:
            article_id = art_match.group(1)
        else:
            article_id = f'section_{i}'

        header = f"[LOC&PAS TSI, section {article_id}] "
        augmented_text = header + piece

        chunks.append({
            'id': f'{doc_type}_{i}',
            'text': augmented_text,
            'metadata': {
                'article': article_id,
                'doc_type': doc_type,
                'subsystem': 'general',
                'chunk_index': i,
                'source_file': pdf_path,
                'language': 'en',
            },
        })

    print(f'  Extracted {len(chunks)} chunks from TSI.')
    return chunks


# ── Generic fallback ──────────────────────────────────────────────────────────

def chunk_generic(pdf_path: str, doc_type: str, language: str = 'en') -> list[dict]:
    """
    Fixed-size chunker for documents without a recognisable heading structure
    (e.g. SpecDoc.pdf). Uses 1500-char chunks with 200-char overlap.
    """
    print(f'  Reading {pdf_path} (generic chunker)...')
    doc = fitz.open(pdf_path)
    full_text = ''.join(page.get_text() for page in doc).strip()

    if not full_text:
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    chunks = []
    step, size = 1300, 1500
    for i, start in enumerate(range(0, len(full_text), step)):
        piece = full_text[start:start + size].strip()
        if len(piece) < 80:
            continue
        chunks.append({
            'id': f'{doc_type}_{i}',
            'text': piece,
            'metadata': {
                'article': f'section_{i}',
                'doc_type': doc_type,
                'subsystem': 'general',
                'chunk_index': i,
                'source_file': pdf_path,
                'language': language,
            },
        })

    print(f'  Extracted {len(chunks)} chunks (generic).')
    return chunks
