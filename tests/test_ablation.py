import pytest

from deletcra.research import ablation_command


def test_embedding_control_changes_only_the_sharing_flag():
    separate = ablation_command("embeddings", "separate", "/data", "/same", 163)
    shared = ablation_command("embeddings", "shared", "/data", "/same", 163)
    assert shared == [*separate, "--share-embeddings"]
    assert separate[separate.index("--mode") + 1] == "rtd"


def test_dropout_control_keeps_time_budget_and_validation_frequency():
    baseline = ablation_command("regularization", "dropout0", "/data", "/same", 163)
    dropout = ablation_command("regularization", "dropout01", "/data", "/same", 163)
    expected = baseline.copy()
    expected[expected.index("--dropout") + 1] = "0.1"
    assert dropout == expected
    assert dropout[dropout.index("--max-training-seconds") + 1] == "163"
    assert dropout[-2:] == ["--eval-every-steps", "500"]


def test_ablation_rejects_unplanned_settings():
    with pytest.raises(ValueError):
        ablation_command("other", "unknown", "/data", "/same", 163)
