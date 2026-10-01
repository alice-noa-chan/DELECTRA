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
See [OPTIMIZATION.md](OPTIMIZATION.md) for verified CUDA loss/gradient checks,
throughput and memory measurements. Completed model training is still pending.

## Optional training execution optimizations

On Linux/CUDA, install the pinned `kernels` extra to use
`--lm-loss-backend liger`. Liger 0.8.4 fuses the existing vocabulary weight and
bias with cross-entropy. It preserves token alignment, valid-pair masking,
LayerNorm, GELU and tied embeddings; unsupported execution raises an error.
CPU evaluation can explicitly use the ordinary torch objective after loading
the same model weights. No Liger-specific model weights are saved.

The fused path projects features into vocabulary logits only at the selected
replacement sites, using features[t-1] to propose x'[t]. The reference logits
sampler also avoids the previous full-sized shifted-logits copy. Generator and
clean main passes skip the unused RTD head, while the corrupted pass keeps it.

`--sequential-backward` is available only for self training. It backpropagates
clean CLM, releases that graph, then accumulates RTD gradients before a single
clip and optimizer update. The returned total retains the clean loss value but
detaches its already-backpropagated graph. Combined and sequential execution
are checked for matching gradients and optimizer updates, including dropout.
`--fused-optimizer` explicitly selects CUDA fused AdamW. Defaults stay unchanged.

```powershell
.venv/Scripts/python -m deletcra run --preset story15m --mode joint --generator-mode self --lm-loss-backend liger --sequential-backward --fused-optimizer --attention-backend flash --device cuda --precision bf16 --data-dir data/tinystories-benchmark --sequence-length 256 --batch-size 64 --steps 300 --probe-steps 0 --output-dir runs/self-liger-new
```

[Liger fused loss API](https://linkedin.github.io/Liger-Kernel/Low-Level-APIs/),
[PyTorch 2.8 fused AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).
