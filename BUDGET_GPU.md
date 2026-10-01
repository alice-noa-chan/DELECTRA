# Budget GPU measurement

The current Runpod console offers one RTX A5000 at $0.16/GPU-hour on October 1,
2026. This is a live offer, not a guaranteed future rate. The public pricing
page previously listed $0.27/hour. Container storage and startup add costs.
Both Community and Secure A5000 stock subsequently ran out before allocation.
The available 24GB RTX 3090 offers $0.22/GPU-hour plus $0.004/container-hour;
it is the fallback for this short measurement. Its result must be labeled 3090,
and cannot be presented as A5000 speed. No A5000 measurement has occurred.

This experiment measures the unchanged 15,041,505-parameter ELECTRA backbone
with integrated learned replacements and RTD + causal LM. It uses native BF16,
forced PyTorch Flash Attention, Liger 0.8.4 fused vocabulary loss, sequential
backward, fused AdamW, context 256, and dropout 0.1. The architecture is unchanged.
The separate generator remains available as the research control.

Use the same pinned 9,999,825-target TinyStories cache as the Modal experiments.
Run three fresh repetitions each at physical batches 64, 128, and 256, rotating
the order. Each trial has 20 warmup and 100 measured optimizer steps. Verify
Flash/eager causal equivalence, Liger loss and gradients, sequential backward,
finite training, and twelve Flash forward/backward operators per joint step.
The Linux runner has a 900-second process deadline and preserves partial evidence
on failure. It does not allocate, extend, or stop the provider's Pod itself.

After committing the code, run in the repository root on one rented A5000:

```sh
python -m deletcra.budget_benchmark --gpu 'RTX 3090' \
  --data-dir data/tinystories-gpu-profile-256-10m-100k \
  --output results/runpod-3090-20261001.json \
  --hourly-price 0.22
```

Pin PyTorch 2.8.0, Transformers 4.57.6, and Liger 0.8.4 to match previous runs.
Export the result before stopping the temporary Pod. A container-only Pod loses
its filesystem on stop; retain evidence locally. Avoid persistent volumes for
this disposable measurement so stopped storage does not keep accruing charges.

The report divides 16.4B prediction targets by measured targets/second and
multiplies GPU hours by the observed GPU price. It counts 255 targets per
256-token block, making the extrapolation slightly conservative relative to an
input-token budget. Complete trials are required before publishing a summary.

This is a cost and throughput experiment, not a full training or model-quality
claim. It excludes preprocessing, startup, storage, failures, evaluation,
TinyStories completion, and instruction tuning. A Base GPU estimate below $15
does not establish that all three finished models cost below $15. Full training
is a separate decision; the accepted 3.3B unique / 16.4B cumulative Base budget
has not been reduced.
