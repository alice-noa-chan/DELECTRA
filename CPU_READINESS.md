# Local CPU preparation and verification

The October 2, 2026 CPU work prepares data and verifies the production training
path before another paid GPU decision. It does not train a release model or
allocate cloud hardware. All three release variants still need their planned
training and quality evaluation before publication.

## Prepared data

| Corpus | Current evidence | Training data |
| --- | --- | --- |
| Full pinned TinyStories | Complete; every file audited | 1,799,248 unique documents; 431,917,550 content tokens |
| September 2026 Common Crawl, prose-v1 | Preparation in progress | Target: at least 3,300,000,000 distinct content tokens |
| Short non-reasoning SmolTalk2 subset | Complete; every token/label file audited | 57,336 examples; 3,243,864 assistant targets |

TinyStories scans **2,141,709 official rows**, rejects 230 empty records and
326,842 normalized exact duplicates across partitions, and retains the full
official training coverage after these explicit exclusions. One prepared train
pass uses **435,417,088 nominal input positions** and **433,716,240 next-token
targets**. The 431,917,550 content-token count excludes inserted record separators
and is measured before 558 discarded train-tail tokens. The proposed 3B-input
allowance would be about 6.89 passes; its adequacy has not been established.

IT scans 151,583 rows from three pinned SmolTalk2 components and retains 58,452
total examples, split into 57,336 train, 531 validation and 585 test examples.
It rejects 92,874 over-context rows and 256 unfinished rows, and removes one
duplicate. Replies are not truncated. One train pass contains 11,750,284 visible
positions but **14,678,016 padded compute positions**. A possible three-epoch
recipe would use 44,034,048 padded positions and 9,731,592 assistant targets;
that is a proposal, not a selected or completed training recipe. The verified
source update is October 2025, not a claim of 2026 conversation collection.

The initial navigation-heavy web baseline is preserved separately at
`data/base-2026-39-256`, with an archive note. The final candidate uses
`data/base-2026-39-prose-256`, a different cleaning identity, and never appends
cleaned text to the baseline. Exact deduplication and crawl language annotations
remain a transparent baseline; near-duplicate removal and learned text-quality
classification are not implemented.

A qualitative inspection of twelve random blocks from an incomplete committed
prefix found readable technical/general prose alongside shopping copy, forms,
name indexes, newsletter/cookie text and disclaimers. Fixed block indices,
source shards and token hashes are recorded in
[the content audit](results/cpu-base-content-audit-20261002.json). This is not a
representative full-corpus quality estimate. It does show that prose-v1 still
retains boilerplate. Completion of the 3.3B-token candidate and its integrity
audit must not be treated as approval for paid full training: review improved
extraction/filtering or a curated source before selecting the final Base data.

Source revisions, licenses, partition rules and resumable commands are in
[TRAINING_PLAN.md](TRAINING_PLAN.md). Ignored local data retains source SHA-256
manifests and pinned upstream dataset cards where available.

## Actual 15M CPU training-path checks

The story diagnostic uses the 15,041,505-parameter production preset, context
256, batch two, FP32 CPU SDPA, self replacement and joint RTD+CLM. A run paused
after three steps resumes to six steps. A separate uninterrupted six-step run
matches its **model tensors, optimizer state, sampler state, Torch RNG and
replacement RNG bit for bit**. Total diagnostic input is 3,072 positions,
not a full TinyStories training run. The resumed run records 14.14 active seconds;
this tiny timing does not project production GPU performance.

The SFT diagnostic loads that same shape-compatible story diagnostic checkpoint
and runs three steps, 1,536 padded positions and 504 assistant targets. This
verifies checkpoint initialization and response-only optimization. It is **not
an IT model derived from a trained Base checkpoint**.

Private archived checkpoints remain in `runs/cpu-production-check`,
`runs/cpu-production-uninterrupted` and `runs/cpu-it-check`. Their frozen diagnostic
scores and source hashes are recorded separately:

- [Story diagnostic](results/cpu-story-diagnostic-test-20261002.json): four blocks,
  PPL 22,179.71; RTD recall zero. Majority-class RTD accuracy is not useful quality
  evidence here.
- [SFT diagnostic](results/cpu-it-diagnostic-test-20261002.json): four blocks,
  response-token PPL 23,283.81.

These deliberately short runs establish execution and resume behavior. Their
poor generation scores do not establish the quality of a fully trained model.

## Frozen reference on the complete new story test split

The pinned `nickypro/tinyllama-15M` revision
`a97e01fa7d54c088df9dc1b53d581a863ce08cd6` scores **NLL 1.757637 / PPL 5.798719**
on all 6,266 blocks and 1,597,830 next-token targets of the new TinyStories test
partition. CPU evaluation takes 980.78 seconds with batch four and two threads.
No model update or checkpoint selection occurs during this scoring.
[Full reference result](results/reference-full-test-cpu-20261002.json).

This is a different document partition from the historical pilot validation
PPL 3.94. Compare future candidates with the 5.80 reference on these identical
test blocks. Reference pretraining overlap with the public corpus is unknown.
Perplexity and full partition coverage alone do not establish story quality.

## Verification and remaining decisions

The complete TinyStories and IT scans verify file SHA-256, all token ID ranges,
packing/count alignment and, for IT, unshifted supervised-label alignment:
[TinyStories audit](results/cpu-stories-integrity-20261002.json),
[IT audit](results/cpu-it-integrity-20261002.json).

The production implementation includes shuffled complete-pass coverage,
token-based warmup/cosine scheduling, exact optimizer/RNG checkpoints,
assistant-only loss and a frozen evaluator. Ruff passes and the CPU suite has
183 passing tests. GPU-only FlashAttention/Liger behavior still needs a bounded
real-corpus GPU pilot after the user chooses the budget.

A later focused run exposed a transient Windows sharing lock during checkpoint
rotation. Saving now keeps the existing latest checkpoint readable while its
previous backup is copied, retries brief sharing locks, and atomically replaces
latest at the end. Injected permanent failure leaves both old copies readable.

Complete and audit the remaining Base data first. Then choose the production
data quality recipe and actual epoch/input allowances, measure the prepared
loader with the
chosen GPU, compare joint self replacement against the separate-generator
reference and CLM control, and complete independent generation evaluation.
No Hugging Face model publication has taken place during this CPU work.
