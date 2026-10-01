# Three evaluated DELECTRA models

The intended Hugging Face lineup contains three trained and evaluated variants.
The existing 300-step TinyStories pilot is archived research evidence, not one of
the publishable models. No pilot upload is authorized by this plan.

| Intended variant | Training | Status |
| --- | --- | --- |
| DELECTRA-15M-TinyStories | Full pinned TinyStories train split, sufficient optimization, story evaluation | Not yet trained on the full corpus |
| DELECTRA-15M-Base | Recent English data, joint RTD+CLM pretraining | Not yet trained |
| DELECTRA-15M-IT | Instruction tuning from the evaluated Base checkpoint | Not yet trained |

Base is the non-instruction-tuned checkpoint. IT is derived from that same Base;
it does not repeat a second independent full pretraining run. CLM-only remains an
internal control for RTD research, not a fourth primary release. A generator is
an auxiliary training artifact, not a fourth inference model either.

## Accepted Base data and training budget

The user chose the original paper's basic ELECTRA-small scale:

- About **3.3B tokens of distinct corpus data** after preparation.
- About **16.4B cumulative input tokens**, with repeats counted.
- Retain the current 15M causal ELECTRA architecture, RTD discriminator, and
  generator/main embedding sharing; use RTD+CLM so the main LM head is trained.
- Keep BF16, forced FlashAttention, physical batch 64 and context 256. The initial
  L40S profile is complete; the repeated GPU comparison now favors RTX PRO 6000
  for cost and H100 for elapsed time. Confirm the chosen GPU with the production
  corpus loader. Batch/LR choices require validation before the full run.

The later [integrated-model optimization](OPTIMIZATION.md) and
[budget GPU measurement](BUDGET_GPU.md) provide additional candidates rather
than changing the accepted token budget or the historical baseline above.
Runpod RTX 3090 self-replacement joint training at batch 256 measured 139,816
targets/s: 32.6 hours / $7.17 GPU-only, or $7.30 including its container, for
16.4B targets. The same small backbone and RTD head are retained. Self replacement
changes the proposal algorithm, so compare quality against the separate generator
before selecting the full-training recipe. The actual fresh-corpus loader,
checkpointing and evaluation still require validation. Full training has not
been started by the short benchmark. The user subsequently requested training,
then deferred allocation until a total-credit review through IT is complete.
See the [whole-lineup scenario](BUDGET_GPU.md#whole-lineup-credit-scenario-october-1-2026)
for proposed TinyStories/IT allowances, storage and evaluation costs, and explicit
unknowns. Its conditional $13-15 range is not an accepted spending limit or a
guaranteed cost to meet the release criteria.

The [original ELECTRA paper](https://cs.stanford.edu/~kevclark/resources/electra.pdf)
uses 3.3B Wikipedia/BooksCorpus tokens for its basic experiments. Basic small
uses 1M steps, batch 128, and length 128: **16.384B nominal sequence positions**.
These include special/padded positions and are not exactly DELECTRA's shifted
next-token target count. New corpus/tokenizer counts must be recorded explicitly.
This matches an approximate data/training scale, not the exact original objective,
architecture, tokenizer, number of epochs, or compute.

The publicly released small corresponds to **small++**, rather than that basic
experiment. Appendix D uses the larger XLNet corpus, 4M steps and length 512 for
small++; the repository identifies its released small with that setting.
[Original repository](https://github.com/google-research/electra).
The accepted plan does not silently adopt this much larger budget.

Sixteen GB of raw text, tokenizer tokens, and repeated training input tokens are
different quantities. We will count tokens, documents, raw UTF-8 bytes, inserted
special tokens, repeats, and discarded tails separately. A 3.3B-token int32 token
array is approximately 13.2GB before manifests and validation; repeated epochs
do not require another copy of the dataset.

## Recent data, with actual dates

Use crawl/capture dates and pinned source revisions, not just repository update
dates, to describe freshness. The primary fresh-source candidate is
**Common Crawl CC-MAIN-2026-39**, the September 2026 crawl. The official index
lists captures from September 4 through September 17, 2026.
[Common Crawl index inventory](https://index.commoncrawl.org/collinfo.json),
[crawl releases](https://commoncrawl.org/blog).

Raw Common Crawl is not a ready-to-train corpus. Build a bounded English subset
with text extraction, language checks, quality filtering, document deduplication,
and independent evaluation partitions before counting the 3.3B tokens. Pin file
lists and capture dates, persist filtering counts, preserve source attribution,
and measure CPU/download/preprocessing costs separately from GPU training.
Freshness does not by itself establish text quality or knowledge accuracy.

FineWeb/FineWeb-Edu are useful curated alternatives and filtering references, but
the verified repositories currently stop at **CC-MAIN-2025-26** (June 2025).
Do not describe them as a 2026 corpus or silently substitute them for the latest
crawl. A mixed-age fallback needs an explicitly documented change of data scope.
[FineWeb card](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
[FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Revision/date observations are saved in the accompanying source inventory.

For IT, inspect recent short English instruction data separately from pretraining.
SmolTalk2 is a candidate with distinct SFT, reasoning/non-reasoning, and preference
components. Select suitable short non-reasoning examples; do not feed its entire
long-context/reasoning mixture into a 15M, context-256 model. Record component
revisions and licenses, chat formatting, and assistant-only loss masks.
[SmolTalk2 card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2).
It was last updated in October 2025 in the verified inventory; this is not a
claim that its conversations were collected in 2026. An IT data recipe remains
to be finalized after the Base and short-example audit.

## Complete TinyStories before considering its release

Prepare the **entire official train split** at the already pinned revision,
without the pilot's 10M-token cap. Record complete document coverage, token
counts, any rejected records, and incomplete packing tails. Never mix validation
or test documents into training. Full corpus coverage is required but one pass
alone is not proof of convergence or quality.

Use a scheduled learning curve with periodic validation and comparable CLM/joint
selection rules. The pilot's main/generator evaluation weights have no optimizer
or RNG resume state. Preserve them; start the production run cleanly unless a
documented warm start is intentionally selected.

Reserve fresh, fixed test documents outside historical pilot evaluation before
checkpoint selection. Score the reference and candidate on identical test tokens
with the same packing and decoding rules, and report selection validation
separately from the final test. Reference pretraining overlap remains unknown.
Use the existing common-validation reference PPL 3.94 as a diagnostic; the pilot
CLM/joint PPL 41.87/47.27 does not meet the target.

## Training and publication sequence

1. Preserve the current pilot, controls, generator, tokenizer, matching
   validation tokens, code, licenses, and raw evidence. This is complete locally;
   originals also remain in the Modal Volume.
2. Implement disk shards/mmap, full-split preparation, token-based LR warmup/decay,
   and optimizer plus CPU/CUDA/RNG/sampler resume states. Verify that interrupted
   training resumes correctly before scheduling multi-day GPU work.
3. Train and evaluate the full TinyStories model; record final coverage, learning
   curves, quality tests, and checkpoint selection. Keep unsuccessful runs private.
4. Prepare/audit the recent corpus, run a longer throughput/convergence pilot,
   and pretrain Base for the accepted 3.3B-data/16.4B-input budget.
5. Derive IT from the selected Base using assistant-response SFT. Fix role
   formatting, EOS behavior, response masking, and stop behavior; freeze the
   evaluated Base checkpoint rather than overwriting it.
6. Evaluate each candidate. TinyStories needs coherent, consistent stories and
   reference-level common-protocol loss; Base needs held-out LM/RTD results and
   readable completions; IT needs held-out instruction compliance and response
   quality compared with its Base. Register concrete pass criteria before final
   testing. Completing a token budget does not automatically pass a release.
7. Only publish candidates whose training and quality checks pass. Prepare
   verified Hugging Face loading/generation, tokenizer/chat templates, safe weight
   export, English model cards, reproducible manifests, MIT additions, and the
   original ELECTRA Apache 2.0 NOTICE/license. No placeholder weights or pilot
   checkpoints should be uploaded as completed models.

The current custom evaluation checkpoints require the deletcra loader and are
not ready-made AutoModelForCausalLM repositories. Native Hub-compatible export
and IT training are still implementation work, not completed capabilities.

## Conditional compute estimate

The repeated fixed-batch comparison measured joint throughput at context 256 and
Flash batch 64. Across two allocations, RTX PRO 6000 projects approximately
**22-25 optimizer hours and $66-76 GPU-only** for 16.4B input targets. Exact H100
projects about **19.5 hours and $77**, while the first-phase L40S and A100 80GB
project **42.1 hours/$82** and **34.9 hours/$87** respectively. RTX is the current
cost candidate; H100 is the time candidate. The longer confirmation's requested
CPU/host-memory estimate adds about $3.47 on RTX and $3.09 on H100.

These are conditional throughput projections, not a completed long training run
or total invoice. They exclude production data preparation, startup, saving,
validation, IT, storage, egress and the separate full TinyStories run. Production
shards, larger context, different batch or CPU bottlenecks require new profiling.
The historical twenty-step L40S estimate of 40.1 hours/$78 remains documented in
[ATTENTION.md](ATTENTION.md); the longer comparison supersedes it for planning.
See [GPU_COMPARISON.md](GPU_COMPARISON.md) for protocols, raw results, pricing,
between-allocation variation and correctness checks. No full training is launched
by this cost experiment.

## Archived pilot

The implementation now also supports integrated self replacement. The bounded
[optimization study](OPTIMIZATION.md) suggests a batch-256 candidate on RTX PRO
6000, but does not establish quality equivalence or change the accepted release
recipe. Compare separate joint, self joint and CLM with matched quality selection
before adopting that variant for full training. Exact resume, full-corpus data
and independent release tests remain required.

The frozen local copy is at
`runs/archives/tinystories15m-20261001-pilot`; it contains 52 copied files and
about 165MB, with per-file SHA-256 hashes. Its manifest explicitly marks
training incomplete and publication false. See
[archive evidence](results/tinystories-pilot-archive-20261001.json).

Reproduce into new paths:

```powershell
.venv/Scripts/python archive_pilot.py --output-dir runs/archives/pilot-new --report results/pilot-archive-new.json
```
