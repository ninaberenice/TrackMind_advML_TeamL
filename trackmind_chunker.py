"""
trackmind_chunker.py
====================
Improved document chunkers for the TrackMind ingestion pipeline.

KEY UPDATE: chunk_generic_from_bytes() added.
  The spec PDF is no longer ingested into ChromaDB. Instead it's uploaded
  per-session and chunked purely in memory. chunk_generic_from_bytes() accepts
  raw PDF bytes (from an uploaded file) and returns chunks as a list of dicts
  without touching the filesystem or the vector database.

The NNTR and TSI chunkers (chunk_french_nntr, chunk_tsi_loc_pas) are unchanged —
they are still used during the one-time ingestion notebook run.

ROOT CAUSE ANALYSIS (unchanged — see original comments)
-------------------
[Bug 1-3 NNTR and TSI chunker fixes are preserved below]
"""

import re
import io
import fitz  # PyMuPDF


# ── NNTR: Arrêté du 19 mars 2012 ─────────────────────────────────────────────

_NNTR_HEADER_RE = re.compile(
    r'28 mars 2012\s*\n'
    r'JOURNAL OFFICIEL DE LA R[EÉ]PUBLIQUE FRAN[CÇ]AISE\s*\n'
    r'Texte 36 sur 102\s*\n'
    r'[.\s]*\n',
    re.IGNORECASE,
)

_NNTR_ART_SPLIT_RE = re.compile(
    r'(?=\nArt\.\s+\d+(?:er|ère|re|nd|ème)?[\w]*\.?\s*[.−\-])'
)

_NNTR_DIVIDERS_RE = re.compile(
    r'(?=\n(?:TITRE|CHAPITRE|Section|ANNEXE)\s+[IVXivx\d]+)'
)

_NNTR_ART_NUM_RE = re.compile(
    r'^Art\.\s+(\d+(?:er|ère|re|nd)?)',
    re.IGNORECASE,
)

_NNTR_DIVIDER_ID_RE = re.compile(
    r'^(TITRE\s+[IVXivx\d]+|CHAPITRE\s+[IVXivx\d]+|Section\s+\d+|ANNEXE\s+[IVXivx\d]+)',
    re.IGNORECASE,
)


def _strip_nntr_headers(text: str) -> str:
    text = _NNTR_HEADER_RE.sub('\n', text)
    text = re.sub(r'\n\s*\.\s*\n\s*\.\s*\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def chunk_french_nntr(pdf_path: str, doc_type: str = 'NNTR_FRANCE') -> list[dict]:
    """
    Article-level chunker for the Arrêté du 19 mars 2012 (French NNTR).
    Reads from filesystem path. Used during one-time ingestion.
    """
    print(f'  Reading {pdf_path}...')
    doc = fitz.open(pdf_path)
    raw_text = ''.join(page.get_text() for page in doc)

    if not raw_text.strip():
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    return _chunk_nntr_text(raw_text, doc_type, source_file=pdf_path)


def _chunk_nntr_text(raw_text: str, doc_type: str, source_file: str = '') -> list[dict]:
    clean = _strip_nntr_headers(raw_text)

    combined_pattern = re.compile(
        r'(?=\nArt\.\s+\d+(?:er|ère|re|nd|ème)?[\w]*\.?\s*[.−\-])'
        r'|(?=\n(?:TITRE|CHAPITRE|Section\s+\d|ANNEXE)\s+[IVXivx\d])'
    )
    raw_chunks = combined_pattern.split(clean)

    chunks = []
    for i, piece in enumerate(raw_chunks):
        piece = piece.strip()

        art_match = _NNTR_ART_NUM_RE.match(piece)
        div_match = _NNTR_DIVIDER_ID_RE.match(piece)

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
                'source_file': source_file,
                'language': 'fr',
            },
        })

    print(f'  Extracted {len(chunks)} chunks from NNTR.')
    return chunks


# ── TSI: LOC&PAS Commission Regulation ───────────────────────────────────────

_TSI_VERSION_STAMP_RE = re.compile(
    r'02014R1302\s*—\s*EN\s*—\s*[\d.]+\s*—\s*[\d.]+\s*—\s*\d+\s*\n?'
)

_TSI_AMEND_MARKERS_RE = re.compile(r'[▼►◄]\s*[MB]\d*\s*|►C\d*\s*')

_TSI_SECTION_ID_RE = re.compile(r'^([1-7](?:\.\d{1,2})+)\.?\s')
_TSI_ARTICLE_ID_RE = re.compile(r'^(Article\s+\d+)', re.IGNORECASE)


def _strip_tsi_noise(text: str) -> str:
    text = _TSI_VERSION_STAMP_RE.sub('', text)
    text = _TSI_AMEND_MARKERS_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def chunk_tsi_loc_pas(pdf_path: str, doc_type: str = 'TSI_LOC_PAS') -> list[dict]:
    """
    Section-level chunker for the LOC&PAS TSI.
    Reads from filesystem path. Used during one-time ingestion.
    """
    print(f'  Reading {pdf_path}...')
    doc = fitz.open(pdf_path)
    raw_text = ''.join(page.get_text() for page in doc)

    if not raw_text.strip():
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    return _chunk_tsi_text(raw_text, doc_type, source_file=pdf_path)


def _chunk_tsi_text(raw_text: str, doc_type: str, source_file: str = '') -> list[dict]:
    clean = _strip_tsi_noise(raw_text)

    combined_pattern = re.compile(
        r'(?=\n[1-7](?:\.\d{1,2})+\.?\s+[A-Z])'
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
            article_id = sec_match.group(1).rstrip(".")
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
                'source_file': source_file,
                'language': 'en',
            },
        })

    print(f'  Extracted {len(chunks)} chunks from TSI.')
    return chunks


# ── Generic chunker (filesystem path) ────────────────────────────────────────

def chunk_generic(pdf_path: str, doc_type: str, language: str = 'en') -> list[dict]:
    """
    Fixed-size chunker for documents without a recognisable heading structure.
    Reads from filesystem path.
    """
    print(f'  Reading {pdf_path} (generic chunker)...')
    doc = fitz.open(pdf_path)
    full_text = ''.join(page.get_text() for page in doc).strip()

    if not full_text:
        print(f'  WARNING: No text extracted from {pdf_path}.')
        return []

    return _chunk_text_to_fixed_size(full_text, doc_type, language, source_file=pdf_path)


# ── Generic chunker (in-memory bytes) — NEW for uploaded spec ────────────────

def _chunk_spec_text(full_text: str, doc_type: str, source_name: str) -> list[dict]:
    # Strip repeated page headers/footers
    lines = full_text.split('\n')
    cleaned_lines = [
        l for l in lines
        if 'IberRail S.A.' not in l
        and 'iberrail.es' not in l
        and 'TECHNICAL SPECIFICATION' not in l
        and 'Passenger Door & Access' not in l
        and 'RESTRICTED' not in l
        and not l.strip().startswith('Page ')
        and not (l.strip().isdigit())
    ]
    full_text = '\n'.join(cleaned_lines)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    combined_pattern = re.compile(
        r'(?=\n\d+\.\d+[\.\d]*\s+[A-Z])'
        r'|(?=\nOI-\d+\s)'
        r'|(?=\nV-\d+\s)'
    )
    raw_chunks = combined_pattern.split(full_text)
    _SEC_ID_RE = re.compile(r'^(\d+\.\d+[\.\d]*)')
    _OI_ID_RE  = re.compile(r'^(OI-\d+)')
    _V_ID_RE   = re.compile(r'^(V-\d+)')
    chunks = []
    for i, piece in enumerate(raw_chunks):
        piece = piece.strip()
        if len(piece) < 200:
            continue
        sec_match = _SEC_ID_RE.match(piece)
        oi_match  = _OI_ID_RE.match(piece)
        v_match   = _V_ID_RE.match(piece)
        if sec_match:
            article_id = sec_match.group(1).rstrip('.')
        elif oi_match:
            article_id = oi_match.group(1)
        elif v_match:
            article_id = v_match.group(1)
        else:
            article_id = f'section_{i}'
        header = f"[{source_name}, {article_id}] "
        chunks.append({
            'id': f'{doc_type}_{i}',
            'text': header + piece,
            'metadata': {
                'article': article_id,
                'doc_type': doc_type,
                'subsystem': 'general',
                'chunk_index': i,
                'source_file': source_name,
                'language': 'en',
            },
        })
    return chunks

def chunk_generic_from_bytes(
    pdf_bytes: bytes,
    doc_type: str = 'SPEC',
    source_name: str = 'uploaded_spec.pdf',
    language: str = 'en',
) -> list[dict]:
    """
    Fixed-size chunker for an uploaded spec PDF given as raw bytes.
    Does NOT write to disk or ChromaDB — output stays in memory.

    This is the function called by api.py's /upload-spec endpoint.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the uploaded PDF file.
    doc_type : str
        Tag stored in chunk metadata (default 'SPEC').
    source_name : str
        Original filename, stored in metadata for display.
    language : str
        Language code (default 'en').

    Returns
    -------
    list[dict]
        Chunk dicts with keys: id, text, metadata.
        Embeddings are NOT pre-computed here — they are computed lazily in
        retrieve_with_session_spec() on first query and cached on the chunk.
    """
    print(f'  Chunking uploaded spec ({source_name}, {len(pdf_bytes)//1024} KB)...')

    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")
    full_text = ''.join(page.get_text() for page in doc).strip()
    doc.close()

    if not full_text:
        print(f'  WARNING: No text extracted from {source_name}.')
        return []

    chunks = _chunk_spec_text(full_text, doc_type, source_name)
    print(f'  Spec ready: {len(chunks)} chunks in memory (not persisted to DB).')
    return chunks


def _chunk_text_to_fixed_size(
    full_text: str,
    doc_type: str,
    language: str,
    source_file: str = '',
    chunk_size: int = 1500,
    step: int = 1300,
) -> list[dict]:
    """
    Shared fixed-size chunking logic used by both chunk_generic() and
    chunk_generic_from_bytes(). 1500-char chunks with 200-char overlap.
    """
    chunks = []
    for i, start in enumerate(range(0, len(full_text), step)):
        piece = full_text[start:start + chunk_size].strip()
        if len(piece) < 80:
            continue
        chunks.append({
            'id': f'{doc_type}_{i}',
            'text': piece,
            'metadata': {
                'article':     f'section_{i}',
                'doc_type':    doc_type,
                'subsystem':   'general',
                'chunk_index': i,
                'source_file': source_file,
                'language':    language,
            },
        })
    return chunks