# Mixed precision and free accelerator continuation

Only free Colab resources are authorized for the current work. Runpod allocation
and automatic paid fallback are excluded.

## CUDA FP16

The production CLI accepts `--precision fp16` for CUDA, including the free T4.
FP32 parameters and Adam moments are retained. Autocast uses FP16 while one
GradScaler, initially 1024, scales both clean CLM and corrupted RTD backward
passes. Gradients are unscaled once before clipping and stepping. The scaler is
saved with the checkpoint and restored on strict resume. BF16/FP32 do not scale
their losses. ELECTRA parameters, detached shifted proposals, RTD labels and
loss weights are unchanged.

Input counters include batches consumed during a skipped FP16 optimizer update.
`optimizer_updates` and `skipped_updates` distinguish these events from the
processed-input budget. Nonfinite forward loss still fails rather than silently
continuing. Old FP32/BF16 checkpoints lacking a scaler remain loadable with their
original configuration.

The [PyTorch AMP guide](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
describes scaling, unscaling before clipping and skipped overflow updates. A CPU
gradient equivalence test verifies that separately scaled backward passes match
scaling the combined loss, including dropout and non-unit objective weights.
Actual T4 FP16 training must be checked separately from this mathematical test.

## XLA and explicit migration

The production CLI now accepts `--device xla --precision bf16 --loss torch`.
Install a compatible Torch/XLA pair in a TPU runtime; importing the ordinary
CPU/CUDA package does not require XLA. This initial implementation targets one
TPU device, uses dense vocabulary logits and fixed-shape masked CE/RTD, and
starts with small physical batches. It is not an optimized multi-TPU trainer.
The [XLA overview](https://docs.pytorch.org/xla/release/r2.9/learn/xla-overview.html)
describes lazy execution, compilation and explicit synchronization.

Host-generated selection/categorical uniforms use a checkpointed CPU generator.
Inverse-CDF categorical sampling evaluates every position in fixed [B,L-1,V]
tensors, excluding special tokens. Only selected eligible tokens are replaced;
the proposal at t comes from logits at t-1 and remains detached. CPU tests cover
CE/gradient equivalence to sparse masking, deterministic shifted replacements,
unchanged-sample labels and separate/self generator gradient paths. The new RNG
algorithm preserves the intended distribution but not legacy random draws.

Validate tokens on CPU before device transfer. Only shape checks run inside the
XLA model; direct XLA callers must supply semantically validated batches. Pad
partial batches with BOS-only rows, exclude them from losses and nominal input
counters, and retain actual compute positions in metrics. No corpus examples are
dropped. AdamW uses device step counters and tensor learning rates on XLA to
avoid changing host scalar constants at every step. Checkpoints materialize all
model/optimizer tensors on CPU and preserve XLA device RNG state separately.
XLA synchronization advances its device seed even without random work. Snapshot
reads therefore do not synchronize; checkpoint saving synchronizes first, and
resume drains loading before restoring the saved next-step seed.

Strict `--resume` still rejects specification changes. `--resume --migrate`
allows device, precision, attention/loss kernel, fused optimizer, host thread
count and process wall limit changes. It preserves model architecture, data
manifest, objective weights, batch size, token budget, sampler cursor and LR
schedule. Adam moments survive the boundary; step tensors move to the target
optimizer's required device. Execution migration resets device/proposal RNG to
`seed + 3 + saved_step`, records both specifications and consumed positions, and
never claims bitwise cross-device continuation. FP16 scaling starts fresh on an
execution migration. Same-configuration resume restores the saved scaler/RNG.

`max_wall_seconds` continues to bound cumulative active training time. To extend
the process allowance after a free-session handoff, explicitly migrate with a
larger allowance; do not accidentally restart warmup or reset input counters.
Each provider transfer must export a complete run directory and verify hashes
before the source runtime disappears. Historical checks in [COLAB.md](COLAB.md)
are hardware/short-run evidence, not long-run quality or throughput evidence.
