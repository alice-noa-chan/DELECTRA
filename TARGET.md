# Target: TinyLlama-15M-level English story generation

The first target is English TinyStories generation. Matching parameter count
alone does not achieve this goal. Compare next-token loss with
`nickypro/tinyllama-15M` on the same validation tokens, and inspect fixed-prompt
stories for sentence flow, consistent characters/events, and repetition.
RTD remains part of the main DELECTRA experiment, with CLM-only as its control.

The publication plan now includes a fully trained TinyStories variant, a Base
variant on recent English data, and an instruction-tuned derivative of Base.
The user selected approximately 3.3B distinct data tokens and 16.4B cumulative
input tokens for Base. Existing pilots are preserved but must not be uploaded
as completed models. See [TRAINING_PLAN.md](TRAINING_PLAN.md) for the corrected
scope, freshness evidence, training sequence, and release requirements.

15M is a reasonable initial scale for this restricted English-story target.
The [TinyStories paper](https://arxiv.org/abs/2305.07759) reports coherent stories
from models below 10M parameters. That supports feasibility, not a guarantee for
this causal ELECTRA architecture or a broad general-purpose language model.
Train the current scale and examine learning curves before attributing the
300-step quality gap to insufficient parameter count.

The tied 32,000 by 288 token table accounts for 9,216,000 parameters; about 5.83M
remain for the other blocks and heads. The reference uses the same token-table
dimensions, so this is a comparison of similar total sizes. Parameter count is
not a substitute for matching data, training budget, and validation protocol.

## Comparison protocol

| Setting | Reference | DELECTRA `story15m` |
| --- | --- | --- |
| Main model parameters | 15,191,712 | 15,041,505 |
| Context length | 256 | 256 |
| Tokenizer / vocabulary | Pinned reference tokenizer, 32,000 entries | Identical |
| Architecture | Llama, RoPE, RMSNorm, SwiGLU | Causal ELECTRA, absolute positions, LayerNorm, GELU |
| Hidden / layers / heads | 288 / 6 / 6 | 288 / 6 / 6 |

This compares similarly sized models with different architectures. The backbone
and RTD head remain ELECTRA. The preset uses embedding width equal to hidden
width, so it does not exploit smaller factorized embeddings. Joint training
shares generator/discriminator token and position embeddings. The generator has
9,487,625 parameters on its own, but the combined unique optimized parameter
count is 15,239,402. Sharing does not remove its additional forward/backward work.

Pinned model revision: `a97e01fa7d54c088df9dc1b53d581a863ce08cd6`.
Pinned TinyStories revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
Official Hugging Face train and validation splits are prepared separately.
Blocks begin with BOS, stories are separated by BOS, and no EOS is appended.
Only full 256-token blocks are used. Each block supplies 255 next-token targets,
excluding its initial BOS as a target.

Overlap between reference pretraining and the official validation split is
unknown. This protocol supports relative comparison, not independent held-out
evidence for the reference. Upstream private-shard loss 1.072 and earlier
WikiText PPL values use different protocols and are not direct thresholds.
Sources: [reference model card](https://huggingface.co/nickypro/tinyllama-15M),
[pinned configuration](https://huggingface.co/nickypro/tinyllama-15M/blob/a97e01fa7d54c088df9dc1b53d581a863ce08cd6/config.json),
[TinyStories card](https://huggingface.co/datasets/roneneldan/TinyStories), and
[llama2.c](https://github.com/karpathy/llama2.c).

## Completed measurements: 2026-10-01

The reference was evaluated on CPU FP32 using 392 validation blocks and 99,960
next-token targets. Its **NLL was 1.371750 and PPL was 3.942243**.
[Reference evidence](results/tinyllama15m-baseline-20261001.json) also preserves
greedy generations from four fixed prompts.

The candidate pilot ran CLM and RTD+CLM on real Modal L40S using BF16, seed 7,
batch 16, dropout 0.1, and 300 steps each. Prepared training data contained
9,999,825 tokens; each condition processed 1,224,000 input tokens. Validation
used the same 99,960 targets as the reference. Learning rate was fixed at 0.0003,
and no frozen probe was trained.

| Condition | Final GPU PPL | Training tokens/s | Training time | Peak allocated |
| --- | --- | --- | --- | --- |
| CLM | 41.87 | 160,749 | 8.27 s | 1.97 GiB |
| RTD+CLM, shared embeddings | 47.27 | 72,591 | 17.45 s | 3.61 GiB |

These are throughput/memory pilots. **Reference-level quality has not been
achieved.** Early training from one seed does not determine RTD's long-run value.
Peak allocated is PyTorch training allocation, not total VRAM. Throughput excludes
timing warmup, validation, saving, and startup.

Saved candidate checkpoints were also evaluated on CPU FP32 using the same
benchmark code: CLM PPL **41.871525**, joint PPL **47.267249**. Small differences
from GPU values arise from evaluation precision. Reference/candidate local
validation file hashes match. The downloaded Modal validation tensor was also
checked token by token and matched exactly. Serialized `.pt` file hashes differ
across PyTorch versions, so canonical decoded int64 token hashes are recorded
as well.

- [Modal reports, provenance, and extrapolation](results/modal-l40s-tinystories15m-20261001-profile.json)
- [Three checkpoint checks and SHA-256](results/modal-l40s-tinystories15m-20261001-profile-checkpoints.json)
- [CLM common benchmark and samples](results/tinystories15m-clm-pilot-benchmark-20261001.json)
- [Joint common benchmark and samples](results/tinystories15m-joint-pilot-benchmark-20261001.json)
- [Modal app](https://modal.com/apps/gaon12/main/ap-kRTMiMqhpFsTCNXqbYPoxK)

The GPU function took 53.86 seconds, yielding a GPU-only estimate of $0.029.
The app stopped; data and checkpoints remain in the `deletcra-experiments`
Volume. All three checkpoints passed strict loading, finite-output checks, and
zero prefix difference after changing a future token. CUDA/BF16 finite-gradient
checks passed too.

## Longer-training budgets

These historical estimates use eager attention and batch 16. The new
[FlashAttention/batch benchmark](ATTENTION.md) measures a separate, faster
execution configuration; its extrapolation remains conditional on sustained
throughput and on validating the larger batch's learning schedule.

The table extrapolates short-pilot throughput for the same model, context, batch,
and GPU. Input budgets are not amounts of unique data, known reference training
budgets, or guarantees of matching reference quality. The recorded GPU rate is
$0.000542/second. [Pricing source](https://modal.com/pricing).

| Input tokens per condition | CLM time / GPU estimate | Joint time / GPU estimate |
| --- | --- | --- |
| 100M | 10.4 min / $0.34 | 23.0 min / $0.75 |
| 1B | 1.73 h / $3.37 | 3.83 h / $7.47 |
| 3B | 5.18 h / $10.12 | 11.48 h / $22.40 |

Validation, saving, startup/scaledown, CPU, host memory, storage, and egress are
excluded. These are not invoice amounts. Sustained throughput may be lower.
Running both conditions or multiple seeds adds the respective costs. No RTX 5090
measurement was made, so these timings do not establish 5090 speed.

The next stage is to **prepare 100M tokens of diverse train data and inspect
learning curves at 100M input tokens per condition before expanding to 1B**.
Current preparation is capped at 100M train tokens in memory. Longer training
needs a warmup/cosine learning-rate schedule and checkpoints containing optimizer
and RNG state for exact resumption. Current model files store evaluation weights,
not full training state. Larger datasets also need shard/mmap support. The budget
table does not mean those longer-run tools or target quality are complete.

Final CLM/joint comparisons should match validation selection procedures across
multiple seeds. RTD-only remains supported, but its main LM projection is
untrained. Evaluating it as a story generator requires an additional CLM stage
whose input and compute budgets must also be recorded.

## Reproduction

```powershell
.venv/Scripts/python -m deletcra prepare --dataset tinystories --sequence-length 256 --max-train-tokens 100000 --max-validation-tokens 100000 --output-dir data/tinystories-benchmark --allow-download
.venv/Scripts/python benchmark_target.py --data-dir data/tinystories-benchmark --output results/reference-new.json --allow-download
.venv/Scripts/python -m modal run target_app.py::target_main --run-id tinystories15m-unique-profile
.venv/Scripts/python benchmark_target.py --data-dir data/tinystories-benchmark --checkpoint runs/<experiment>/clm/model --output results/candidate-new.json
```

Output paths must be new. Use the same validation file for reference and
candidate, checked by the JSON file hash. The recorded measurements also include
a decoded-token hash for cross-version verification. Generation uses four fixed
prompts, greedy decoding, and at most 96 new tokens. Candidate generation masks
BOS/PAD; the reference uses native Hugging Face generation. These samples are
qualitative, not a controlled generation-quality score. This decoding difference
does not affect the perplexity comparison.

New code is MIT-licensed. Original ELECTRA's Apache 2.0 attribution remains in
NOTICE and LICENSES. The reference card's MIT label and TinyStories'
CDLA-Sharing-1.0 apply separately. Model and dataset files are not committed.
