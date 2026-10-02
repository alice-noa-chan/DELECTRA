import json

import pytest
import torch

from deletcra.runtime_probe import main, probe_runtime


def test_cpu_probe_checks_gradients_without_training_claim():
    report = probe_runtime("cpu")
    assert report["status"] == "passed"
    assert report["actual_device"] == "cpu"
    assert report["checks"]["fp32"]["status"] == "passed"
    assert report["delectra_training_verified"] is False


def test_requested_cuda_never_silently_uses_cpu(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("DELECTRA_PROBE_DEVICE", "cuda")
    destination = tmp_path / "probe.json"
    monkeypatch.setenv("DELECTRA_PROBE_OUTPUT", str(destination))
    with pytest.raises(SystemExit, match="1"):
        main()
    report = json.loads(destination.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["delectra_training_verified"] is False


def test_probe_rejects_unknown_device():
    with pytest.raises(ValueError, match="requested device"):
        probe_runtime("automatic")
