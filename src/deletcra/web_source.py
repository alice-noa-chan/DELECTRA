"""Filter and tokenize one WET source on a CPU core, without importing Torch."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from tokenizers import Tokenizer
from warcio.archiveiterator import ArchiveIterator

from deletcra.web_text import clean_web_text, document_key, quality_reason


def prepare_source(path: Path, output: Path, tokenizer_path: Path):
    """Stage candidates; the main process owns global deduplication and commits.

    Workers may finish out of order. A bounded Arrow file returns IDs and hashes,
    not an entire in-memory corpus. The caller consumes these files in original
    source order, so scheduling cannot change partitions or deduplication winners.
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    schema = pa.schema(
        [
            ("hash", pa.binary(32)),
            ("utf8_bytes", pa.int64()),
            ("ids", pa.list_(pa.int32())),
        ]
    )
    pending = []
    counts = Counter()
    started = time.monotonic()
    temporary = output.with_suffix(".partial")
    with temporary.open("wb") as sink, ipc.new_file(sink, schema) as writer:

        def flush():
            if not pending:
                return
            encoded = tokenizer.encode_batch(
                [text for _, text in pending], add_special_tokens=False
            )
            rows = [
                {
                    "hash": key,
                    "utf8_bytes": len(text.encode("utf-8")),
                    "ids": encoding.ids,
                }
                for (key, text), encoding in zip(pending, encoded, strict=True)
            ]
            writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
            pending.clear()

        with path.open("rb") as archive:
            for record in ArchiveIterator(archive):
                if record.rec_type != "conversion":
                    continue
                counts["seen_documents"] += 1
                language = (
                    record.rec_headers.get_header("WARC-Identified-Content-Language")
                    or ""
                )
                if language.strip() != "eng":
                    counts["rejected_language"] += 1
                    continue
                raw = record.content_stream().read().decode("utf-8", errors="replace")
                if not raw.strip():
                    counts["empty"] += 1
                    continue
                text = clean_web_text(raw)
                counts["removed_nonprose_utf8_bytes"] += len(raw.encode("utf-8")) - len(
                    text.encode("utf-8")
                )
                reason = quality_reason(text, language)
                if reason:
                    counts["rejected_" + reason] += 1
                    continue
                pending.append((document_key(text), text))
                counts["candidate_documents"] += 1
                if len(pending) == 512:
                    flush()
        flush()
    temporary.replace(output)
    return {
        "counts": dict(counts),
        "processing_seconds": time.monotonic() - started,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_source(args.input, args.output, args.tokenizer)))


if __name__ == "__main__":
    main()
