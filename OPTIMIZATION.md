# Integrated-training optimization evidence

Measurements use the unchanged 15,041,505-parameter ELECTRA-derived story preset:
six layers, hidden/embedding width 288, vocabulary 32,000, context 256, dropout
0.1, BF16 and forced native CUDA FlashAttention. RTD retains its original head,
learned replacements, actual-change labels and a second corrupted-input pass.
See [MODEL_REVIEW.md](MODEL_REVIEW.md) for the model critique and quality limits.

## Initial six-case experiment

The exact device was NVIDIA RTX PRO 6000 Blackwell Server Edition, running
PyTorch 2.8.0+cu128 and CUDA 12.8. The image pins Liger 0.8.4. Code:
`46f29b80e253ef5eee44b33836411f623d7fb2e9`.
[Modal execution](https://modal.com/apps/gaon12/main/ap-yzn4IXhkm4fqhX3eTRc85s),
[raw evidence](results/modal-integration-20261001-profile.json).

Each case uses three fresh trials, 20 warmup and 200 measured complete optimizer
steps, including data selection/transfer, corruption, backward, gradient clipping,
finite checks and AdamW. Case order rotates across trials. Each trial resets model
and sampling seeds. LR is 0.0003 and weight decay 0.01. The input is the existing
pinned 10M-token TinyStories cache; this is a throughput test, not full training.
Reported tokens are shifted targets: batch times 255 per step. Throughput is total
measured targets divided by total optimizer seconds. Peak allocated GiB excludes
other processes, CUDA context and non-PyTorch allocations.

| Joint training configuration | Batch | Targets/s | Peak allocated GiB | Trial throughput SD |
| --- | ---: | ---: | ---: | ---: |
| Separate generator, shared embeddings, torch loss | 64 | 198,496 | 10.45 | 1.36% |
| Self replacement, torch loss | 64 | 301,896 | 6.25 | 1.26% |
| Self replacement, Liger | 64 | 305,407 | 2.56 | 0.09% |
| Self + Liger + sequential backward | 64 | 296,223 | 2.33 | 0.60% |
| Self + Liger + sequential + fused AdamW | 64 | 295,900 | 2.33 | 0.48% |
| Self + Liger + sequential + fused AdamW | 256 | 390,735 | 8.52 | 0.66% |

At batch 64, integration raises throughput 52.1% versus the current separate
baseline. Liger removes another 59.1% of self-mode allocated memory, with only
1.2% higher throughput. Sequential backward saves another 9.0% of Liger memory
but runs about 3.0% slower here. Fused AdamW shows no measurable speed advantage.
Keep both flags optional; memory savings are not automatically speed savings.

The measured batch-256 configuration is 1.97 times the separate batch-64
baseline's throughput and uses 18.4% less allocated memory. Batch 256 uses fewer
updates per target and self replacement changes the proposal distribution;
neither change establishes equal convergence or final model quality.
Historical GPU comparisons used earlier sampler/head execution and different
allocations. Use this run's own baseline for optimization ratios.

The 16.4B-target arithmetic projection is 22.95 hours / $69.57 GPU-only for the
current separate baseline, or 11.66 hours / $35.34 for measured self batch 256.
These are throughput extrapolations, not completion promises. They exclude data
preparation, evaluation, checkpointing, host, storage, startup and retries, and
assume unchanged throughput. Original ELECTRA's nominal sequence-position budget
is not identical to this shifted-target count. Do not replace the release recipe
until matched quality experiments justify it.

## Correctness gates and cost

The FP32 masked/shifted Liger loss differs by 4.77e-7; weight, bias and feature
gradient maximum differences are at most 2.98e-8. BF16 loss differs by 0.00384;
the largest relative gradient L2 error is 0.419%, with cosine similarity at least
0.999993. These are small finite-precision differences, not bitwise equivalence.
CUDA combined/sequential execution produced identical replacements and total
loss, with maximum gradient difference 1.34e-4. CPU tests additionally compare
one clipped optimizer update and multistep saved weights.

Profiler counts prove 14 native FlashAttention forward/backward operations for
separate joint and 12 for self joint. Every measured step checks finite loss and
gradients. All 18 trials completed, totaling 88,128,000 measured targets.

GPU function time was 333.72 seconds, giving **$0.281 GPU-only** at the previously
verified Modal rate of $0.000842/second. This is an execution estimate, not an
invoice. Requested CPU/host memory, image build, startup and storage are separate.
The completed app was verified stopped with zero running tasks. An earlier
Windows console-encoding failure created an empty app, which was also stopped.

## Follow-up experiment

The initial results motivate a bounded, separate comparison of the measured
batch-256 configuration against ordinary combined backward/AdamW at batch 256
and batch 512. It uses three fresh trials, 20 warmup and 100 measured steps,
rotating all three settings with a same-allocation control. Its shorter protocol
is recorded separately rather than merged with the initial measurements.

```powershell
.venv/Scripts/python -X utf8 -m modal run optimization_app.py --run-id YOUR_NEW_RUN_ID --follow-up
```

No finished checkpoints or Hugging Face models are produced by these tests.
