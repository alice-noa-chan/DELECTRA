"""Pure text normalization, partitioning and prose filters for CPU workers."""

import hashlib
import re
import unicodedata
from collections import Counter


def document_key(text: str) -> bytes:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def web_split(key: bytes) -> str:
    """Assign each content hash once: 98% train, 1% validation, 1% test."""
    bucket = int.from_bytes(key[:8], "big") % 10000
    return "validation" if bucket < 100 else "test" if bucket < 200 else "train"


def clean_web_text(text: str) -> str:
    """Keep prose lines and discard short menu/link labels before token counting.

    WET already extracts plaintext, but retains navigation and footer text.
    Keep lines with ten words and terminal sentence punctuation, or twenty words
    regardless of punctuation. Deduplicate identical lines within a document.
    These transparent rules may remove useful headings and poetry; they are
    intended for this small English prose model, not general web preservation.
    """
    kept, seen = [], set()
    for raw in text.splitlines():
        line = " ".join(raw.split())
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line)
        if len(words) < 10 or (
            len(words) < 20 and not line.endswith((".", "!", "?", '"'))
        ):
            continue
        key = document_key(line)
        if key not in seen:
            seen.add(key)
            kept.append(line)
    return "\n".join(kept)


def quality_reason(text: str, languages: str) -> str | None:
    """Transparent baseline filtering, not a FineWeb-quality classifier.

    Require English-only crawl annotations, readable alphabetic text, common
    English function words, and varied prose. Exact normalized-document hashing
    handles repeats; near-duplicate removal is not established by this filter.
    """
    if languages.strip() != "eng":
        return "language"
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not 100 <= len(words) <= 20000:
        return "length"
    # C-backed split/map operations preserve the same Unicode predicates while
    # avoiding Python generator overhead for every character in a large crawl.
    nonspace = len("".join(text.split()))
    if not nonspace or sum(map(str.isalpha, text)) / nonspace < 0.7:
        return "alphabetic_ratio"
    lower = [word.lower() for word in words]
    frequencies = Counter(lower)
    if len(frequencies) / len(lower) < 0.1:
        return "repetition"
    function_words = {"the", "a", "an", "and", "of", "to", "is", "in", "for", "that"}
    if sum(frequencies[word] for word in function_words) / len(lower) < 0.03:
        return "english_function_words"
    return None
