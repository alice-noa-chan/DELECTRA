import json

import pytest

from deletcra.corpus import MmapSplit
from deletcra.prepare_cpu import Preparation, document_key, quality_reason, web_split


class FakeTokenizer:
    unk_token = "unk"

    def save_pretrained(self, path):
        path.mkdir(exist_ok=True)

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[2 + len(text) % 5] * 300 for text in texts]}


@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "deletcra.prepare_cpu.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )


def test_preparation_resume_dedup_and_rollback(tmp_path, fake_tokenizer):
    prep = Preparation(tmp_path, {"source": "fixture"})
    source = {"id": "first"}
    prep.unit(source, iter([("A story", "train", ""), (" a  STORY ", "test", "")]))
    assert prep.manifest["counts"]["exact_duplicates"] == 1
    assert len(MmapSplit(tmp_path, "train")) == 1

    def broken_rows():
        yield "second story", "train", ""
        raise RuntimeError("interrupted source")

    with pytest.raises(RuntimeError, match="interrupted"):
        prep.unit({"id": "second"}, broken_rows())
    prep.db.close()
    resumed = Preparation(tmp_path, {"source": "fixture"})
    resumed.unit(source, iter([("must not replay", "train", "")]))
    resumed.unit({"id": "second"}, iter([("second story", "train", "")]))
    resumed.complete()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["counts"]["train_documents"] == 2
    assert len(metadata["units"]) == 2
    assert len(MmapSplit(tmp_path, "train")) == 2
    with pytest.raises(ValueError, match="identity"):
        Preparation(tmp_path, {"source": "changed"})


def test_content_partition_and_quality_filters():
    assert document_key("A\n story") == document_key("a story")
    assert web_split(document_key("A story")) in {"train", "validation", "test"}
    text = " ".join(
        ["The book is in the library and a child reads it for a friend."] * 8
    )
    assert quality_reason(text, "fra") == "language"
    assert quality_reason("The book", "eng") == "length"
    # A repetitive fixture fails; language tags alone are insufficient.
    assert quality_reason(text * 100, "eng") == "repetition"


def test_prefetch_preserves_order_and_download_path_guard(tmp_path, monkeypatch):
    from deletcra.prepare_cpu import download_wet, prefetched_wet

    monkeypatch.setattr(
        "deletcra.prepare_cpu.download_wet", lambda cache, name: (cache / name, name)
    )
    result = list(prefetched_wet(["third", "first", "second"], tmp_path, workers=2))
    assert [name for name, _ in result] == ["third", "first", "second"]
    with pytest.raises(ValueError, match="unexpected crawl"):
        download_wet(tmp_path, "../unrelated")


def test_web_cleaning_removes_navigation_and_deduplicates_prose():
    from deletcra.prepare_cpu import clean_web_text

    prose = "The small town library opens each morning and welcomes every local child."
    raw = f"Home\nSkip Navigation\nPrivacy policy\n{prose}\n{prose}\nContact us"
    assert clean_web_text(raw) == prose


def test_failed_download_uses_alternate_endpoint_and_atomic_cache(
    tmp_path, monkeypatch
):
    import hashlib

    import requests

    from deletcra.prepare_cpu import download_wet

    calls = []
    delays = []

    class Download:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            if len(calls) == 1:
                raise requests.HTTPError("503")

        def iter_content(self, size):
            yield b"fixture compressed bytes"

    def get(url, **kwargs):
        calls.append(url)
        assert not list(tmp_path.glob("*.wet.gz"))
        return Download()

    monkeypatch.setattr("deletcra.prepare_cpu.requests.get", get)
    monkeypatch.setattr("deletcra.prepare_cpu.time.sleep", delays.append)
    name = "crawl-data/CC-MAIN-2026-39/fixture.wet.gz"
    path, digest = download_wet(tmp_path, name)
    assert calls == [
        "https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/" + name,
        "https://data.commoncrawl.org/" + name,
    ]
    assert delays == [5]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.partial"))
    assert download_wet(tmp_path, name) == (path, digest)
    assert len(calls) == 2


def test_encoded_web_rejects_unknown_without_poisoning_resume_dedup(
    tmp_path, fake_tokenizer
):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    prep = Preparation(tmp_path / "corpus", {"source": "fixture"})
    candidate_file = tmp_path / "candidates.ipc"
    invalid_key = document_key("literal unknown")
    valid_key = document_key("valid story")
    schema = pa.schema(
        [
            ("hash", pa.binary(32)),
            ("ids", pa.list_(pa.int32())),
            ("utf8_bytes", pa.int64()),
        ]
    )
    rows = [
        {"hash": invalid_key, "ids": [2, 0, 3], "utf8_bytes": 16},
        {"hash": valid_key, "ids": [2, 3] * 200, "utf8_bytes": 400},
    ]
    with candidate_file.open("wb") as handle, ipc.new_file(handle, schema) as writer:
        writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=schema))
    stats = {
        "counts": {"seen_documents": 2, "candidate_documents": 2},
        "processing_seconds": 1,
    }
    prep.unit_encoded({"id": "first"}, candidate_file, stats, 10000)
    assert prep.manifest["counts"]["rejected_unknown_tokens"] == 1
    assert prep.manifest["counts"].get("unused_candidate_documents", 0) == 0
    assert (
        prep.db.execute("SELECT 1 FROM docs WHERE hash=?", (invalid_key,)).fetchone()
        is None
    )
    prep.db.close()
    resumed = Preparation(tmp_path / "corpus", {"source": "fixture"})
    resumed.unit_encoded({"id": "second"}, candidate_file, stats, 10000)
    resumed.complete()
    assert resumed.manifest["counts"]["rejected_unknown_tokens"] == 2
    assert resumed.manifest["counts"]["exact_duplicates"] == 1
    assert (
        sum(
            resumed.manifest["counts"].get(f"{s}_documents", 0)
            for s in ("train", "validation", "test")
        )
        == 1
    )
