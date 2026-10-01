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

For an available A5000, use `--gpu 'RTX A5000'`, a new output filename, and its
actual hourly offer. Never reuse the 3090 result as an A5000 measurement.

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

## Completed RTX 3090 measurement

The [raw result](results/runpod-3090-20261001.json) contains nine complete trials,
34,272,000 measured prediction targets, and source commit
`f5612fef95ada57e6f0f41255467d4f2d3b57646`. All three original Modal cache files
matched SHA-256 after transfer. The driver reported NVIDIA GeForce RTX 3090,
24GB VRAM, driver 580.178.04, and PyTorch 2.8.0+cu128. The recorded
[package inventory](results/runpod-3090-20261001-packages.txt),
[benchmark log](results/runpod-3090-20261001.log), and
[provenance](results/budget-gpu-provenance-20261001.json) preserve the evidence.

| Physical batch | Targets/s | Trial speed SD | Peak allocated GiB | Peak reserved GiB | Base hours | GPU only | GPU + 30GB container |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 110,578 | 1.04% | 2.33 | 5.26 | 41.20 | $9.06 | $9.23 |
| 128 | 130,193 | 1.93% | 4.39 | 10.12 | 34.99 | $7.70 | $7.84 |
| 256 | 139,816 | 0.34% | 8.51 | 22.21 | 32.58 | $7.17 | $7.30 |

Rates use total targets divided by total measured seconds across three trials.
Batch 256 was 26.4% faster than batch 64 and 7.4% faster than batch 128.
Allocator reservation was much larger than live allocation; this experiment
does not establish that batch 512 fits or improves speed on this card.

Flash/eager BF16 checks passed, and changed future tokens produced exactly zero
prefix difference. Liger FP32 loss difference was below 0.000001; the largest
BF16 gradient relative L2 error was 0.419%, with cosine similarity above 0.99999.
Sequential and combined backward produced identical replacements and losses;
maximum gradient difference was 0.000134. Every profiled training step dispatched
twelve native Flash forward and twelve backward operations, with finite losses
and clipped gradients.

The complete benchmark process took 341.50 seconds, approximately $0.0213 at
the GPU + container rate. This excludes container initialization, installation,
transfer and operator time. The Pod showed 15m 50s uptime at the stop request:
approximately $0.059 at the quoted rate, not a provider invoice. Results and
environment files were exported and their archive/file hashes verified before
the user-approved container deletion. The Pod is **stopped**, with both compute
and container storage reported **Not running**, and **$0.00/hour** total.

At the measured batch-256 speed, the accepted Base token budget is plausibly
below $15 for GPU and this container. Even half the measured throughput would
project about 65.2 hours / $14.60 for those two items. Neither projection includes
fresh-corpus filtering, full-corpus loading overhead, quality pilots, validation,
checkpoints, failures, TinyStories completion or IT. Confirm production-loader
speed and the current offer before allocating full training. No finished model
or quality improvement was demonstrated by this short benchmark.
