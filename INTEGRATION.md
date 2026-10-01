# Integrated causal ELECTRA research

`--mode joint --generator-mode self` uses one ELECTRA backbone with its original
RTD head and the existing tied LM head. The clean pass supplies both next-token
supervision and detached replacement proposals from position t-1. A second pass
classifies the corrupted input. It retains learned replacements, actual-change
labels, all eligible-token RTD supervision, causal attention, absolute positions,
LayerNorm and GELU. This is a self-replacement variant, not original ELECTRA.

The main CLM loss is counted once. There is no independent generator loss or
generator checkpoint. RTD gradients reach the shared backbone but cannot travel
through discrete replacement sampling into the LM projection. A supplied auxiliary
generator or embedding-sharing flag is rejected instead of silently ignored.

```powershell
.venv/Scripts/python -m deletcra run --mode joint --generator-mode self --steps 100 --probe-steps 0 --output-dir runs/self-smoke-new
```

Separate-generator training remains the default and the historical baseline.
Integration changes the proposal distribution and gradient balance, so equal
token budgets do not establish equal quality. Track actual replacement rate,
majority baseline, balanced accuracy, AUROC/AP and held-out LM loss. Use matched
compute and validation selection to assess the contribution of RTD.

Original ELECTRA investigated full weight tying, but preferred a small generator
with shared embeddings for efficiency. DELECTRA already requires a clean main
CLM pass, making proposal reuse a different tradeoff.
[ELECTRA paper, section 3.2](https://cs.stanford.edu/~kevclark/resources/electra.pdf).

Implementation checks cover two backbone passes, correct loss weighting,
detached sampling, future isolation, RTD/LM gradients and single-model export.
This document does not claim measured CUDA speed or completed model training.
