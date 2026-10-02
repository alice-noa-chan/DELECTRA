import numpy as np
import pytest

from deletcra.audit import audit_corpus
from deletcra.corpus import TokenWriter, file_sha256, write_json


def pretraining_fixture(directory):
    writer = TokenWriter(directory / "train.bin", 4, 10)
    writer.append([2, 3], 1)
    entry = writer.finish()
    return {
        "status": "complete",
        "sequence_length": 4,
        "vocab_size": 10,
        "counts": {"train_content_tokens": 2, "train_documents": 1},
        "splits": {"train": [entry]},
    }


def test_audit_checks_all_pretraining_bytes_and_count_alignment(tmp_path):
    metadata = pretraining_fixture(tmp_path)
    write_json(tmp_path / "metadata.json", metadata)
    result = audit_corpus(tmp_path)
    assert result["splits"]["train"]["prediction_targets"] == 3
    assert result["splits"]["train"]["input_positions"] == 4
    (tmp_path / "train.bin").write_bytes(np.array([2, 4, 1], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="hash"):
        audit_corpus(tmp_path)


@pytest.mark.parametrize("invalid", ["incomplete", "counts", "path", "token"])
def test_audit_rejects_invalid_pretraining_artifacts(tmp_path, invalid):
    metadata = pretraining_fixture(tmp_path)
    if invalid == "incomplete":
        metadata["status"] = "preparing"
    elif invalid == "counts":
        metadata["counts"]["train_documents"] = 2
    elif invalid == "path":
        metadata["splits"]["train"][0]["path"] = "../escape.bin"
    else:
        (tmp_path / "train.bin").write_bytes(np.array([2, 0, 1], dtype="<i4").tobytes())
    write_json(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError):
        audit_corpus(tmp_path)


def test_audit_checks_sft_labels_even_when_hashes_match(tmp_path):
    ids = np.array([1, 3, 2, 0], dtype="<i4")
    labels = np.array([-100, 3, 2, -100], dtype="<i4")
    tokens_path, labels_path = tmp_path / "tokens.bin", tmp_path / "labels.bin"
    tokens_path.write_bytes(ids.tobytes())
    labels_path.write_bytes(labels.tobytes())
    split = {
        "examples": 1,
        "visible_positions": 3,
        "padded_compute_positions": 4,
        "assistant_targets": 2,
        "tokens": {"path": "tokens.bin", "sha256": file_sha256(tokens_path)},
        "labels": {"path": "labels.bin", "sha256": file_sha256(labels_path)},
    }
    metadata = {
        "status": "complete",
        "kind": "assistant-only-sft",
        "sequence_length": 4,
        "vocab_size": 10,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "splits": {"train": split},
    }
    write_json(tmp_path / "metadata.json", metadata)
    assert audit_corpus(tmp_path)["splits"]["train"]["assistant_targets"] == 2
    labels[1] = 4
    labels_path.write_bytes(labels.tobytes())
    split["labels"]["sha256"] = file_sha256(labels_path)
    write_json(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="differs"):
        audit_corpus(tmp_path)


@pytest.mark.parametrize("ids", [[3, 4, 2, 0], [1, 0, 4, 2]])
def test_sft_audit_rejects_bad_bos_or_interior_padding(tmp_path, ids):
    tokens, labels = tmp_path / "tokens.bin", tmp_path / "labels.bin"
    tokens.write_bytes(np.array(ids, dtype="<i4").tobytes())
    labels.write_bytes(np.array([-100, -100, -100, -100], dtype="<i4").tobytes())
    write_json(
        tmp_path / "metadata.json",
        {
            "status": "complete",
            "kind": "assistant-only-sft",
            "sequence_length": 4,
            "vocab_size": 10,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "splits": {
                "train": {
                    "examples": 1,
                    "visible_positions": 3,
                    "padded_compute_positions": 4,
                    "assistant_targets": 0,
                    "tokens": {"path": tokens.name, "sha256": file_sha256(tokens)},
                    "labels": {"path": labels.name, "sha256": file_sha256(labels)},
                }
            },
        },
    )
    with pytest.raises(ValueError, match="BOS|right"):
        audit_corpus(tmp_path)
