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

## Measured partial-training result

Both requested phases completed. The TPU applied updates 1–256 and the T4
continued updates 257–512 after an exact model/Adam/cursor transfer. No updates
were skipped. The run consumed 2,097,152 input positions, 2,088,960 next-token
targets and 8,192 distinct global blocks in epoch zero: approximately 0.482% of
the first full pass. This is block coverage, not a claim about unique word types
or completed stories. The final and selected-best weights both come from step
512; the full working schedule still has 433,319,936 positions remaining.

| Checkpoint | Device | Cumulative input positions | Validation NLL |
| --- | --- | --- | --- |
| Initialization | TPU BF16 | 0 | 10.452616 |
| 64 | TPU BF16 | 262,144 | 8.811468 |
| 128 | TPU BF16 | 524,288 | 6.369901 |
| 192 | TPU BF16 | 786,432 | 5.491644 |
| 256 | TPU BF16 | 1,048,576 | 4.544028 |
| 320 | T4 FP16 | 1,310,720 | 4.120146 |
| 384 | T4 FP16 | 1,572,864 | 3.917611 |
| 448 | T4 FP16 | 1,835,008 | 3.763245 |
| 512 | T4 FP16 | 2,097,152 | 3.611341 |

These are the same first 128 validation blocks (32,640 LM targets), not the full
validation partition or an independent test. Native BF16/eager and FP16/SDPA
evaluation differ across the boundary. A frozen FP32 CPU check returned
NLL 3.611345 / PPL 37.015822 on the same blocks, confirming the loss result with
an additional precision/backend. The nickypro full-test PPL is from a different
partition and must not be compared directly with this limited validation score.

The frozen RTD check used CPU proposals with seed 1007, actual-change labels and
the usual logit >= 0 threshold. It returned TP 80, TN 28,400, FP 36 and FN 3,974:
precision 68.97%, recall 1.97%, accuracy 87.66%. Always predicting original would
already achieve 87.52% on these labels. High accuracy therefore does not show a
useful detector, and ranking/calibration quality was not measured. Causal RTD's
benefit over a matched CLM control remains unestablished.

All three predetermined greedy prompts exhibit repetition; one changes a rabbit
into a girl mid-continuation. These are unselected diagnostic samples rather
than a generation-quality pass. The checkpoint remains an early research run;
more training, RTD diagnostics and controlled comparisons are needed before any
claim about the intended TinyStories release.

Trainer active time was 279.55 seconds on TPU and another 45.82 seconds on T4,
including in-loop compilation/validation/checkpoint overhead. This excludes
allocation, installation, initial baseline evaluation, transfers and teardown,
and is not a matched sustainable-throughput benchmark or full-training estimate.
The T4 peak allocated CUDA memory was 2,123,208,704 bytes (about 1.98 GiB), which
excludes driver/context and other process memory.

Each 64-update archive and latest checkpoint was SHA-256 verified locally.
Phase-boundary archives also preserve `best.pt`; no weights were pushed to Git
or Hugging Face. Both runtimes were explicitly stopped after export, and final
inventory, account assignment count and usage rate were zero with a 0.00 CU
balance. The overall record links the recovered checkpoints and provenance:
[partial training record](results/colab-stories-partial-training-20261002.json),
[frozen validation](results/colab-stories-frozen-validation-20261002.json),
[generation diagnostics](results/colab-stories-generation-20261002.json),
[cleanup](results/colab-stories-cleanup-20261002.json).
