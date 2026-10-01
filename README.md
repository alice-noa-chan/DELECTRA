# DELECTRA

DELECTRA studies ELECTRA with causal attention and replaced-token detection
(RTD). The backbone and discriminator head come from Hugging Face Transformers
ELECTRA. The main experiment keeps RTD and adds next-token prediction for story
generation; a CLM-only run provides the control for measuring RTD's contribution.

The existing Python package and CLI are named `deletcra`. The repository folder,
saved experiment IDs, and historical artifact paths also retain that spelling.

## Install and develop

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m ruff format .
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest
```

With `uv`, run `uv sync --locked --extra dev`. Add `--extra data` for dataset
preparation and `--extra cloud` for Modal. Choose a CUDA PyTorch wheel that
supports the GPU when training locally.

Follow this order for each coherent change: edit code or documentation, format
and lint, test, then commit. Split independent changes into separate commits.
Write detailed English commit messages with the change, its reason, and the
validation performed. Documentation, references, footnotes, comments, and
docstrings are written in English. Explain important assumptions in plain words.
See [AGENTS.md](AGENTS.md) for project guidance and [HISTORY.md](HISTORY.md) for
the authorized history cleanup and experiment source-hash correspondence.

## Code layout

| Module | Responsibility |
| --- | --- |
| `config.py` | Validate model settings and derive a smaller generator |
| `model.py` | Make ELECTRA causal, expose RTD/LM heads, share embeddings, save/load |
| `attention.py` | Reuse ELECTRA Q/K/V weights with SDPA or forced CUDA FlashAttention |
| `attention_benchmark.py` | Measure real-objective GPU throughput and physical batch memory |
| `objectives.py` | Predict next tokens, sample replacements, and compute losses |
| `data.py` | Pack separate dataset splits into BOS-prefixed token blocks |
| `experiment.py` | Train, validate, measure throughput, and save checkpoints |
| `metrics.py` | Compute RTD ranking metrics with explicit tie handling |
| `benchmark.py` | Score reference and candidate models on the same token targets |
| `target.py` | Pin the TinyStories reference and define the 15M model preset |
| `research.py`, `cloud.py` | Build bounded experiment commands and validate requests |
| `cli.py` | Expose data preparation and local training commands |

Root-level Modal apps run bounded GPU experiments. Analysis scripts read saved
results; they do not launch training. Tests live in `tests/`, and tracked raw
evidence lives in `results/`. Downloaded data and checkpoints live in ignored
`data/` and `runs/` directories.

## ELECTRA structure and learning objective

`CausalElectra` uses Transformers' `ElectraModel` with `is_decoder=True`,
cross-attention disabled, and caching disabled. It retains ELECTRA's learned
absolute positions, LayerNorm, GELU, and discriminator head. The vocabulary head
projects hidden states to embedding width and shares its output weights with the
input token embeddings. Embedding width can differ from hidden width.

At position `t`, RTD judges the token at `t`; LM logits predict the token at
`t+1`. Future tokens cannot affect either head's prefix outputs. Inputs must use
right padding and have a visible first token. Training blocks begin with BOS.
Greedy generation requires unpadded prefixes of equal length.

```python
import torch
from deletcra import ModelConfig
from deletcra.model import CausalElectra

model = CausalElectra(ModelConfig())
output = model(torch.tensor([[1, 4, 5, 6]]))
model.save("runs/example")
restored = CausalElectra.load("runs/example")
```

`CausalElectra.from_encoder_checkpoint(path, bos_token_id=101)` converts a local
ELECTRA discriminator checkpoint. Set BOS to the real tokenizer's CLS/BOS ID.
The backbone and discriminator weights are preserved, while the new LM
projection is untrained. Conversion alone does not produce a trained generator.
Downloads are disabled by default. Our reported training experiments start from
fresh initialization, rather than converted encoder checkpoints.

| Mode | Generator objective | Main model objective |
| --- | --- | --- |
| `clm` | No generator | Clean-input next-token cross entropy |
| `rtd` | Causal next-token cross entropy | RTD on corrupted input |
| `joint` | Causal next-token cross entropy | RTD plus clean-input next-token loss |

Default loss weights are generator 1, RTD 5, and main CLM 1. These are adjustable;
they do not reproduce every setting of the original ELECTRA paper. RTD is
computed at all eligible content positions, including unchanged tokens. BOS,
PAD, and configured special tokens are excluded from replacement and detection.
A selected position receives an original label if the sampled token matches it.

The generator predicts replacement `x'_t` from its logits at `t-1`, conditioned on
the **original prefix**. Sampling is parallel and detached from autograd. The
discriminator sees the corrupted prefix. This differs from sequential generation
conditioned on earlier replacements and from original ELECTRA's bidirectional
MLM generator. The generator learns through its own CLM loss, not adversarial
gradients through sampling.

Joint training uses separate corrupted RTD and clean CLM backbone passes. Equal
input-token budgets therefore do not imply equal compute. Optional
`--share-embeddings` ties generator/discriminator token and position embeddings.
Construct the optimizer after sharing; shared parameters are updated once.
Generator CLM gradients then also train the shared discriminator embeddings.
The TinyStories joint pilot enables sharing; historical independent-embedding
experiments retain their original settings.

### Attention execution backends

`--attention-backend eager` remains the default for historical reproducibility.
`sdpa` reuses ELECTRA's Q/K/V weights through PyTorch scaled-dot-product attention.
`flash` forces PyTorch's CUDA FlashAttention kernel and raises if it cannot run;
it does not silently fall back. No external `flash-attn` package is required.
The installed Transformers ELECTRA has only an eager attention implementation,
so DELECTRA supplies a small adapter while retaining ELECTRA's other blocks.

Flash mode currently requires unpadded blocks and CUDA FP16/BF16 Q/K/V. SDPA also
supports right-padded inputs with a combined causal/padding mask. Both preserve
attention dropout during training and explicitly disable it during evaluation.
Use `CausalElectra.load(path, attention_backend="sdpa")` to inspect flash-trained
weights on CPU. Attention weights, head masks, cross-attention, and caching are
outside this adapter's supported interface. State-dict names and parameter
counts remain unchanged.

See [ATTENTION.md](ATTENTION.md) for real L40S kernel verification, batch-size
measurements, memory limits, and the distinction between throughput and quality.

## Data preparation

The default smoke test uses synthetic cyclic sequences without downloads. Train
and validation starts use different seeds, but share a simple grammar. This
checks optimization rather than real-language generalization.

WikiText-2/103 preparation streams the raw train and validation splits separately.
CLS becomes block BOS, SEP separates nonempty records, and the incomplete tail
is discarded. The test split is untouched. Tokenizer and metadata are saved with
the token tensors. Downloads require `--allow-download`, and existing output
directories are never overwritten.

```powershell
.venv/Scripts/python -m pip install -e ".[dev,data]"
.venv/Scripts/python -m deletcra prepare --output-dir data/wikitext2 --allow-download
```

TinyStories preparation pins the reference tokenizer and dataset revisions. Its
tokenizer uses BOS to separate stories because it has no SEP token. Use context
256 explicitly for the target protocol; see [TARGET.md](TARGET.md) for commands.

Data licenses are separate from the code license. WikiText's official card has
inconsistent CC BY-SA version labels between metadata and prose; preparation
records that source notice. TinyStories is labeled CDLA-Sharing-1.0. Preserve the
upstream dataset notices when distributing data.

## Run and interpret evaluations

Run all three objectives without network access:

```powershell
.venv/Scripts/python -m deletcra run --output-dir runs/smoke --mode all --steps 300 --probe-steps 100
```

Each mode saves `report.json`, `model/`, an optional `generator/`, and optional
`probe.pt`. The CLI also saves `summary.json`. `--train-tokens` replaces a step
budget and rounds up to a full batch. `--max-training-seconds` adds a measured
training-time limit and stops after a completed optimizer step.

CLM and joint report next-token loss and perplexity. RTD-only leaves the main LM
projection untrained, so its main perplexity and generated text are not measures
of a trained language model. A common optional probe trains an identically
initialized linear next-token head on each frozen backbone. That measures
representation usefulness; it does not turn RTD-only into a trained generator.
Use `--probe-steps 0` to skip it.

RTD reports F1, recall, balanced accuracy, majority baselines, AUROC, and average
precision. Track replacement rate as well: the generator changes during
training, so initial and final corruptions need not match. A fixed-generator
diagnostic is used for controlled comparisons in [RESEARCH.md](RESEARCH.md).

Throughput measures training after a short timing warmup and excludes validation,
probe training, and saving. This warmup is for measurement, not a learning-rate
schedule. Timed runs measure all steps. Peak CUDA allocated memory is PyTorch's
training allocation, not total device memory. Selected `best_model` checkpoints
use validation loss; they are not independent test results. Model checkpoints
currently store evaluation weights, not optimizer/RNG state for exact resumption.

## Historical CPU smoke results: 2026-10-01

The offline smoke run used Python 3.11.9, PyTorch 2.14.1+cpu, Transformers 4.57.6,
and one CPU thread. Main/generator sizes were 20,641/2,353 parameters. Each mode
used seed 7, 300 steps, batch 16, length 16, and 72,000 input tokens. Validation
contained 960 next-token targets; each frozen probe trained for 100 steps.
[Raw CPU results](results/cpu-smoke-2026-10-01.json) preserve settings and histories.

| Mode | Initial -> final LM PPL | Frozen probe PPL | Training time |
| --- | --- | --- | --- |
| CLM | 16.20 -> 1.366 | 1.208 | 1.64 s |
| Causal RTD | Main LM projection untrained | 12.014 | 2.43 s |
| Joint | 16.20 -> 1.364 | 1.279 | 3.63 s |

Default RTD accuracy 97.6% matched the majority baseline, F1 was 0, and balanced
accuracy was 0.5. Replacement rate fell from 14.3% to 2.4%. A separate control
with replacement probability 0.5, temperature 5, and 1,000 steps reached F1
0.600, balanced accuracy 0.689, and accuracy 0.720 versus a 0.586 majority
baseline. Multiple settings changed, so this is not a matched comparison or
evidence of superiority on real language.

## Modal GPU experiments

Authenticated Modal runs prepare data on CPU before allocating one L40S 48GB.
The measured results are L40S results, not RTX 5090 benchmarks. Source and license
files are uploaded; local Git metadata, environments, and credentials are not.
Data and checkpoints persist in the `deletcra-experiments` Volume.

```powershell
.venv/Scripts/python -m pip install -e ".[dev,cloud]"
.venv/Scripts/python -m modal run modal_app.py --run-id modal-wikitext2-unique-pilot
.venv/Scripts/python -m modal run modal_app.py --run-id modal-wikitext2-unique-10m --train-tokens 10000000 --probe-steps 100
```

The default pilot runs 100 steps per objective, batch 32, context 128, BF16, and
20 probe steps. Explicit token budgets are capped at 10M per objective. The GPU
function has a 900-second timeout, no automatic retries, and a two-second
scaledown window. CUDA/BF16 prefix isolation and finite gradients are checked
before training. Local reports are saved in `results/<run-id>.json`.

Download checkpoints into an existing parent directory:

```powershell
New-Item -ItemType Directory -Path runs/modal-checkpoints -Force | Out-Null
.venv/Scripts/python -m modal volume get deletcra-experiments /runs/<run-id> runs/modal-checkpoints
```

GPU runtime estimates use the recorded official rate and are not invoices. CPU,
host memory, startup/scaledown, storage, egress, failed setup, and credits are
separate. See the [GPU guide](https://modal.com/docs/guide/gpu) and
[pricing](https://modal.com/pricing). Predict runtime from the same measured
model, context, batch, and GPU; tiny CPU smoke speeds do not predict GPU speeds.

## Historical WikiText-2 GPU results: 2026-10-01

The initial 100-step pilot and 10M-input-token comparison completed on L40S,
CUDA 12.8, PyTorch 2.8.0+cu128, and Transformers 4.57.6. Main/generator sizes
were 13,563,579/4,175,227 parameters. The prepared dataset contained about 1M
train and 100K validation tokens. **The 10M training budget repeatedly sampled
that 1M-token subset; it was not 10M of new unique data.** Each mode used seed 7,
2,461 steps, batch 32, context 128, and 10,001,504 actual input tokens.

This initial evaluation covered only the first 128 validation blocks, or 16,256
next-token targets. Probes trained for 100 steps. Later full-validation results
have different coverage and should not be compared directly with this table.

| Mode | Main PPL | Probe PPL | RTD F1 @ 0.5 | Training s | Tokens/s | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- |
| CLM | 327.06 | 601.24 | N/A | 111.44 | 90,624 | 1.97 |
| Causal RTD | Untrained LM | 3,174.95 | 0.000 | 150.64 | 66,436 | 1.88 |
| Joint | 277.41 | 522.89 | 0.115 | 242.44 | 41,262 | 3.66 |

RTD predicted every token as original at threshold 0.5: accuracy 86.11% equaled
the majority baseline, balanced accuracy was 0.5, and replacement rate was
13.89%. Joint balanced accuracy was 0.523 and recall was 0.069. F1 at one
threshold does not summarize all ranking performance.

Joint PPL was 15.2% lower in this seed but required 2.18 times the training time.
This is neither an equal-compute comparison nor statistical evidence of
superiority. Follow-up experiments address time matching, multiple seeds,
embedding sharing, and regularization.

The comparison's training time totaled 504.52 s, and GPU function time was
533.56 s. The 100-step pilot function took 34.36 s. At $0.000542 per GPU second,
GPU-only estimates were $0.289 and $0.019, or $0.308 combined. The longer run was
slower than the first short-pilot extrapolation. Both project apps stopped;
checkpoints were downloaded and verified for strict loading, finite outputs,
prefix isolation, and bounded CLM/joint generation. Hashes and provenance are
preserved in the raw results.

- [Initial GPU pilot](results/modal-l40s-wikitext2-20261001-pilot.json)
- [10M comparison and checkpoint verification](results/modal-l40s-wikitext2-20261001-10m.json)

## Follow-up research and generation target

[RESEARCH.md](RESEARCH.md) records the three-seed matrix and exploratory
ablations. With full 99,949-target validation and 10M input tokens, joint mean PPL
was 318.98 +/- 3.78 versus CLM 370.62 +/- 3.30. Time-matched dropout-free CLM
overfit, so its final PPL does not establish joint compute efficiency.

In the later seed-7 control, dropout 0.1 with validation checkpoint selection
reached CLM PPL 294.48, below that seed's joint final PPL 314.72. Selection,
regularization, and input budgets were not fully matched. RTD's intrinsic
advantage remains unproven. Shared RTD embeddings improved probe PPL from about
3,238 to 2,128, still well behind CLM/joint.

[TARGET.md](TARGET.md) defines the current English TinyStories generation goal,
the measured TinyLlama-15M reference PPL of 3.94, and real 15M-model throughput
and budget estimates. The candidate runs are 300-step pilots and have not reached
reference-level quality. RTD remains part of the main DELECTRA experiment; CLM
remains its control.

## Licenses and attribution

New code in this repository is [MIT-licensed](LICENSE). Original Google Research
[ELECTRA](https://github.com/google-research/electra) was released under
**Apache 2.0**. [NOTICE](NOTICE) preserves attribution and describes the changes;
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) preserves the upstream license.
MIT for our additions does not replace upstream licenses. External dependencies,
downloaded checkpoints, and datasets retain their respective licenses.
