# Partial TinyStories training on free Colab

The October 2 continuation starts a new tied 15,041,505-parameter DELECTRA run
from random initialization, preserving the ELECTRA backbone and joint self RTD/
CLM objective. Earlier short pilots remain archived and are not added to this
run's input counters. Runpod and paid fallback are excluded.

The working schedule covers one complete pass of the audited pinned TinyStories
training corpus, with context 256, batch 16, LR 0.0003, 1,048,576 warmup positions,
cosine decay and AdamW. One pass is a resumable initial schedule, not a claim
that the final language-quality target needs only one pass. The current request
pauses after 1,048,576 positions on TPU BF16 and another 1,048,576 on T4 FP16.
Checkpoint and validation intervals are 64 updates; validation uses 128 fixed
blocks from the validation split. No independent test data is accessed.

## Transfer windows preserve the full-corpus sampler

`IndexedSplit` exposes the original complete manifest and split length. Its
separate cache manifest stores only requested global block indices and hashed
token files. The preparation protocol copies the first 8,192 blocks of the
original full-corpus shuffle (sampler seed 9), not the first 8,192 documents or a
new shuffle of a smaller dataset. This avoids transferring every corpus shard
for a short phase. A missing block fails before an update rather than being
substituted or skipped. Cache completion never means full-corpus coverage.

The source mmap reader verifies the full shard hashes before export. The remote
reader verifies the transfer-window manifest, global indices and token hashes.
Both devices consume consecutive portions of the same shuffle, retain the same
batch/budget/LR schedule, and preserve Adam moments and the cursor during the
explicit execution migration. Device/proposal RNG is reseeded at the boundary;
cross-device bitwise continuation is not claimed.

The next session can provide another window or the complete `MmapSplit` and
resume the existing specification. A CPU test compares window-to-full-corpus
resume with uninterrupted training, including exact model/Adam state. This
checks loader continuity, not full-trained model quality.

## Preservation and scope

Export and SHA-256-verify durable checkpoints after each bounded training chunk
and before releasing either runtime. Keep the full working budget in every
checkpoint and pause explicitly; do not reset warmup at the device switch.
The cumulative active-training wall allowance is 3,600 seconds. Runtime loss or
free-quota rejection ends the phase at its last recovered checkpoint; no paid
replacement is rented. Reports distinguish requested from reached stop steps.

The plan, corpus provenance and cache fingerprints are in
[the training-window record](results/colab-stories-window-plan-20261002.json).
This partial run remains private and cannot be published as a finished model.
