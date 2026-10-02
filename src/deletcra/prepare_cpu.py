"""Resumable local CPU preparation of pinned stories and fresh web text.

SQLite commits document hashes and source-unit manifests together. Token files
are flushed before that commit. After interruption, only the incomplete unit is
replayed; completed sources, exact deduplication, and split assignments persist.
"""

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from deletcra.corpus import TokenWriter, file_sha256, write_json
from deletcra.target import (
    REFERENCE_MODEL,
    REFERENCE_REVISION,
    STORIES_DATASET,
    STORIES_REVISION,
)


def document_key(text: str) -> bytes:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def web_split(key: bytes) -> str:
    """Assign each content hash once: 98% train, 1% validation, 1% test."""
    bucket = int.from_bytes(key[:8], "big") % 10000
    return "validation" if bucket < 100 else "test" if bucket < 200 else "train"


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
    nonspace = sum(not c.isspace() for c in text)
    if not nonspace or sum(c.isalpha() for c in text) / nonspace < 0.7:
        return "alphabetic_ratio"
    lower = [word.lower() for word in words]
    if len(set(lower)) / len(lower) < 0.1:
        return "repetition"
    function_words = {"the", "a", "an", "and", "of", "to", "is", "in", "for", "that"}
    if sum(word in function_words for word in lower) / len(lower) < 0.03:
        return "english_function_words"
    return None


class Preparation:
    def __init__(self, directory: Path, identity: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "preparation.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS docs (hash BLOB PRIMARY KEY)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, json TEXT)"
        )
        self.db.commit()
        row = self.db.execute("SELECT json FROM state WHERE id=1").fetchone()
        if row:
            self.manifest = json.loads(row[0])
            if self.manifest["identity"] != identity:
                raise ValueError(
                    "resume identity differs from the existing preparation"
                )
        else:
            self.manifest = {
                "identity": identity,
                "status": "preparing",
                "sequence_length": 256,
                "vocab_size": 32000,
                "bos_token_id": 1,
                "pad_token_id": 0,
                "splits": {name: [] for name in ("train", "validation", "test")},
                "units": [],
                "counts": {},
                "deduplication": (
                    "NFKC/casefold/whitespace normalized SHA-256 across splits; "
                    "exact only"
                ),
                "packing": (
                    "BOS prefix per block; BOS record separator; "
                    "per-unit tails discarded"
                ),
            }
        # Never trust a completed-unit marker without its bytes. Recovery needs
        # the token files as well as SQLite; copying only metadata is insufficient.
        for entries in self.manifest["splits"].values():
            for entry in entries:
                if file_sha256(directory / entry["path"]) != entry["sha256"]:
                    raise ValueError(
                        "completed preparation shard failed hash verification"
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(
            REFERENCE_MODEL, revision=REFERENCE_REVISION
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(directory / "metadata.json", self.manifest)

    def unit(self, source: dict, rows, *, web_budget: int | None = None):
        if any(unit["source"]["id"] == source["id"] for unit in self.manifest["units"]):
            return
        index = len(self.manifest["units"])
        counts = Counter()
        writers = {
            split: TokenWriter(self.directory / f"{split}-{index:05d}.bin", 256, 32000)
            for split in self.manifest["splits"]
        }
        self.db.execute("BEGIN")
        pending = []

        def flush():
            if not pending:
                return
            encoded = self.tokenizer(
                [text for _, text in pending], add_special_tokens=False
            )["input_ids"]
            for (split, _), ids in zip(pending, encoded, strict=True):
                if 0 in ids:
                    raise ValueError("reference tokenizer produced UNK/PAD")
                writers[split].append(ids, 1)
                counts[f"{split}_content_tokens"] += len(ids)
            pending.clear()

        started = time.monotonic()
        try:
            for row_number, (text, split, language) in enumerate(rows):
                counts["seen_documents"] += 1
                if not text.strip():
                    counts["empty"] += 1
                    continue
                if web_budget is not None:
                    reason = quality_reason(text, language)
                    if reason:
                        counts["rejected_" + reason] += 1
                        continue
                key = document_key(text)
                if web_budget is not None:
                    split = web_split(key)
                inserted = self.db.execute(
                    "INSERT OR IGNORE INTO docs VALUES (?)", (key,)
                ).rowcount
                if not inserted:
                    counts["exact_duplicates"] += 1
                    continue
                counts[f"{split}_documents"] += 1
                counts["accepted_utf8_bytes"] += len(text.encode("utf-8"))
                pending.append((split, text))
                if len(pending) >= 512:
                    flush()
                    if counts["train_documents"] % 10000 < 512:
                        print(
                            json.dumps(
                                {
                                    "unit": index,
                                    "seen": row_number + 1,
                                    "train_content_tokens": counts[
                                        "train_content_tokens"
                                    ],
                                    "seconds": round(time.monotonic() - started, 1),
                                }
                            ),
                            flush=True,
                        )
                    if web_budget is not None and (
                        self.manifest["counts"].get("train_content_tokens", 0)
                        + counts["train_content_tokens"]
                        >= web_budget
                    ):
                        counts["budget_stop_inside_source"] = 1
                        break
            flush()
            entries = {split: writer.finish() for split, writer in writers.items()}
            candidate = json.loads(json.dumps(self.manifest))
            for split, entry in entries.items():
                candidate["splits"][split].append(entry)
            candidate["units"].append(
                {
                    "source": source,
                    "counts": dict(counts),
                    "seconds": time.monotonic() - started,
                }
            )
            candidate["counts"] = dict(Counter(candidate["counts"]) + counts)
            self.db.execute(
                "INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(candidate),)
            )
            self.db.commit()
            self.manifest = candidate
            write_json(self.directory / "metadata.json", self.manifest)
            print(
                json.dumps({"committed_unit": index, "counts": dict(counts)}),
                flush=True,
            )
        except BaseException:
            self.db.rollback()
            for writer in writers.values():
                if not writer.source.closed:
                    writer.source.close()
            raise

    def complete(self):
        self.manifest["status"] = "complete"
        self.db.execute(
            "INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(self.manifest),)
        )
        self.db.commit()
        write_json(self.directory / "metadata.json", self.manifest)
        self.db.close()


def stories(directory: Path):
    prep = Preparation(
        directory,
        {
            "dataset": STORIES_DATASET,
            "dataset_revision": STORIES_REVISION,
            "tokenizer": REFERENCE_MODEL,
            "tokenizer_revision": REFERENCE_REVISION,
            "license": "CDLA-Sharing-1.0; preserve upstream attribution",
            "validation_partition": (
                "first 2000 official validation rows plus even hashes; "
                "remaining odd hashes are test"
            ),
        },
    )
    info = HfApi().dataset_info(STORIES_DATASET, revision=STORIES_REVISION)
    files = sorted(
        f.rfilename for f in info.siblings if f.rfilename.endswith(".parquet")
    )
    for name in files:
        path = Path(
            hf_hub_download(
                STORIES_DATASET, name, repo_type="dataset", revision=STORIES_REVISION
            )
        )
        official_split = "train" if "/train-" in name else "validation"

        def rows(path=path, official_split=official_split):
            row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=1024, columns=["text"]
            ):
                for text in batch.column(0).to_pylist():
                    split = official_split
                    if official_split == "validation" and row_index >= 2000:
                        if document_key(text)[0] % 2:
                            split = "test"
                    row_index += 1
                    yield text, split, ""

        prep.unit(
            {
                "id": name,
                "sha256": file_sha256(path),
                "official_rows": pq.ParquetFile(path).metadata.num_rows,
            },
            rows(),
        )
    prep.complete()


def download_wet(cache: Path, name: str):
    """Cache one complete source download, retaining its exact-byte hash."""
    if not name.startswith("crawl-data/CC-MAIN-2026-39/") or ".." in name:
        raise ValueError("unexpected crawl source path")
    cache.mkdir(exist_ok=True)
    path = cache / (hashlib.sha256(name.encode()).hexdigest() + ".wet.gz")
    if path.exists():
        return path, file_sha256(path)
    partial = path.with_suffix(".partial")
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with requests.get(
                "https://data.commoncrawl.org/" + name, stream=True, timeout=(30, 120)
            ) as download:
                download.raise_for_status()
                with partial.open("wb") as target:
                    for chunk in download.iter_content(1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
            partial.replace(path)
            return path, digest.hexdigest()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def prefetched_wet(paths, cache: Path, *, workers: int = 4):
    """Overlap up to four source downloads without changing processing order."""
    pending = deque()
    iterator = iter(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in iterator:
            pending.append((name, pool.submit(download_wet, cache, name)))
            if len(pending) == workers:
                break
        while pending:
            name, future = pending.popleft()
            yield name, future.result()
            next_name = next(iterator, None)
            if next_name is not None:
                pending.append((next_name, pool.submit(download_wet, cache, next_name)))


def commoncrawl(directory: Path, budget: int):
    from warcio.archiveiterator import ArchiveIterator

    crawl = "CC-MAIN-2026-39"
    response = requests.get(
        f"https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz", timeout=60
    )
    response.raise_for_status()
    inventory = gzip.decompress(response.content)
    paths = inventory.decode().splitlines()
    # Hash-sort sources to avoid selecting only the first crawl time/domain band.
    paths.sort(key=lambda value: hashlib.sha256(value.encode()).digest())
    prep = Preparation(
        directory,
        {
            "dataset": crawl,
            "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
            "tokenizer": REFERENCE_MODEL,
            "tokenizer_revision": REFERENCE_REVISION,
            "target_unique_train_content_tokens": budget,
            "capture_dates": "2026-09-04 through 2026-09-17",
            "license": (
                "Common Crawl terms; original pages retain their individual rights"
            ),
            "filter": (
                "English-only crawl annotation; 100-20000 words; alphabetic >=0.7; "
                "unique words >=0.1; function words >=0.03"
            ),
            "source_order": "SHA-256 sorted WET paths",
        },
    )
    (directory / "wet.paths.txt").write_bytes(inventory)
    completed = {unit["source"]["id"] for unit in prep.manifest["units"]}
    remaining_paths = [name for name in paths if name not in completed]
    for name, (path, digest) in prefetched_wet(
        remaining_paths, directory / "downloads"
    ):
        if prep.manifest["counts"].get("train_content_tokens", 0) >= budget:
            break

        def rows(path=path):
            with path.open("rb") as archive:
                for record in ArchiveIterator(archive):
                    if record.rec_type != "conversion":
                        continue
                    language = (
                        record.rec_headers.get_header(
                            "WARC-Identified-Content-Language"
                        )
                        or ""
                    )
                    text = (
                        record.content_stream().read().decode("utf-8", errors="replace")
                    )
                    yield text, "train", language

        prep.unit(
            {
                "id": name,
                "sha256": digest,
                "compressed_bytes": path.stat().st_size,
            },
            rows(),
            web_budget=budget,
        )
        # Shards and source hashes are durable. Retain only pending raw sources.
        path.unlink()
    if prep.manifest["counts"].get("train_content_tokens", 0) < budget:
        raise RuntimeError("crawl exhausted before the requested content-token budget")
    prep.complete()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["stories", "commoncrawl"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-content-tokens", type=int, default=3_300_000_000)
    args = parser.parse_args()
    if args.train_content_tokens < 1:
        parser.error("train content budget must be positive")
    if args.source == "stories":
        stories(args.output)
    else:
        commoncrawl(args.output, args.train_content_tokens)


if __name__ == "__main__":
    main()
