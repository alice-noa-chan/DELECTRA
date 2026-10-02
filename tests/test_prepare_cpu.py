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
