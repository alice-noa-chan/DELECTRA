import io
import re

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from warcio.warcwriter import WARCWriter

from deletcra.prepare_cpu import Preparation
from deletcra.web_source import prepare_source


def test_worker_matches_serial_tokens_and_global_dedup(tmp_path, monkeypatch):
    words = ["word" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(300)]
    text = "The " + " ".join(words) + " and of to is in for that a an."
    raw = "Home\nSkip Navigation\n" + text
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    for word in re.findall(r"\w+|[^\w\s]", text):
        if word not in vocab:
            vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    monkeypatch.setattr(
        "deletcra.prepare_cpu.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    wet = tmp_path / "fixture.wet.gz"
    source_rows = [(raw, "eng"), (raw + "\nContact us", "eng"), (raw, "spa")]
    with wet.open("wb") as handle:
        writer = WARCWriter(handle, gzip=True)
        for content, language in source_rows:
            record = writer.create_warc_record(
                "https://example.org/fixture",
                "conversion",
                payload=io.BytesIO(content.encode()),
                warc_headers_dict={"WARC-Identified-Content-Language": language},
            )
            writer.write_record(record)
    serial = Preparation(tmp_path / "serial", {"fixture": True})
    serial.unit(
        {"id": "fixture"},
        ((content, "train", language) for content, language in source_rows),
        web_budget=10000,
    )
    serial.complete()
    parallel = Preparation(tmp_path / "parallel", {"fixture": True})
    candidates = tmp_path / "candidates.ipc"
    stats = prepare_source(
        wet, candidates, tmp_path / "parallel/tokenizer/tokenizer.json"
    )
    assert stats["counts"]["candidate_documents"] == 2
    parallel.unit_encoded({"id": "fixture"}, candidates, stats, 10000)
    parallel.complete()
    for split in ("train", "validation", "test"):
        left = serial.manifest["splits"][split][0]
        right = parallel.manifest["splits"][split][0]
        assert left == right
    assert parallel.manifest["counts"]["exact_duplicates"] == 1
    assert parallel.manifest["counts"]["rejected_language"] == 1
