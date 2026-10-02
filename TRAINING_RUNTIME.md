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

XLA training and explicit device migration will be recorded here after their
implementation and verification. Earlier hardware checks in [COLAB.md](COLAB.md)
do not establish full model training or portable resume on TPU.
