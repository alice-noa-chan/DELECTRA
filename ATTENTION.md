# FlashAttention and batch sizing on L40S

This experiment measures the existing 15M causal ELECTRA with optimized attention
and larger physical batches. It changes how attention is computed, preserving
ELECTRA Q/K/V weights, learned absolute positions, residual connections,
LayerNorm, GELU, the discriminator, and the RTD objective. No parameters are added
or removed. It does not measure story quality.

## Implementation and correctness

Transformers 4.57.6 ELECTRA exposes only eager attention. The adapter in
`src/deletcra/attention.py` reuses its original Q/K/V modules and state-dict names.
`sdpa` allows PyTorch to choose a scaled-dot-product attention kernel; `flash`
forces `SDPBackend.FLASH_ATTENTION` and raises if unsupported. It uses native
PyTorch FlashAttention, without an external `flash-attn` installation.

Fully packed blocks use `is_causal=True` without an explicit attention mask.
Flash mode requires unpadded CUDA FP16/BF16 Q/K/V. Padded inputs are supported by
the SDPA backend with a combined causal and visible-key mask. Training keeps the
original attention dropout; evaluation explicitly sets attention dropout to zero.
See the [PyTorch 2.8 SDPA documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).

The CPU suite checks eager/SDPA outputs and gradients, future/padding isolation,
loss gradient paths for RTD/CLM/joint, checkpoint compatibility, and evaluation
dropout. Actual CUDA BF16 checks compare eager/flash hidden states and both heads
at identical weights, then change a future token and require zero prefix change.
Finite losses and gradients are required throughout every measured training case.

The separate profiler step requires six native Flash forward and backward calls
for CLM, or fourteen of each for joint: six for clean CLM, six for corrupted RTD,
and two for the generator. This verifies the generator uses Flash as well.

## Measurement protocol

- GPU: one NVIDIA L40S; PyTorch 2.8.0+cu128, CUDA 12.8.
- Model: `story15m`, 15,041,505 main parameters; joint has 15,239,402 unique
  optimized parameters after sharing token/position embeddings.
- Data: existing pinned TinyStories cache, about 10M train tokens, full 256-token
  blocks without padding. Batches sample this cache; this is not new training data.
- Precision: CUDA BF16 autocast with FP32 parameters; dropout 0.1; seed 7.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.01, gradient clipping 1.0.
- Objectives: clean CLM, or generator CLM plus corrupted RTD plus clean main CLM.
  Joint uses the project default loss weights 1/5/1 and replacement probability 0.15.
- Timing: five warmup steps, then twenty measured complete optimizer steps per
  case, including batch transfer, loss checks, backward, clipping, and AdamW.
  A separate profiled step is excluded. No validation or checkpoint save is timed.
- Tokens/s: batch size times 255 next-token positions times twenty, divided by
  measured time. Extra generator/backbone passes are not additional input tokens.
- Memory: peak PyTorch allocated and reserved bytes after warmup; neither is total
  VRAM. Each case creates fresh models/optimizer and releases CUDA cache afterward.

The first sweep tested batches 16/32/64/128. A second sweep repeats those controls
and adds 192/256. Out-of-memory results stop larger batches for that condition.
Both runs are bounded, use no automatic retries, and save JSON evidence.

## Measured results: 2026-10-01

The extended sweep completed 21 cases and recorded two out-of-memory cases.
The table uses that single run for consistent comparisons. Throughput is in
thousands of input tokens per second; memory is peak allocated GiB.

| Backend | Batch | CLM k tokens/s | CLM GiB | Joint k tokens/s | Joint GiB |
| --- | --- | --- | --- | --- | --- |
| eager | 16 | 168.8 | 1.97 | 72.7 | 3.61 |
| eager | 32 | 192.8 | 3.71 | 87.6 | 6.96 |
| eager | 64 | 191.2 | 7.20 | 86.2 | 13.70 |
| eager | 128 | 189.6 | 14.18 | 84.6 | 27.16 |
| eager | 192 | 188.1 | 21.16 | OOM | OOM |
| eager | 256 | 186.3 | 28.14 | Skipped after OOM | - |
| flash | 16 | 170.2 | 1.73 | 79.1 | 3.02 |
| flash | 32 | 221.2 | 3.21 | 101.4 | 5.82 |
| flash | 64 | 253.4 | 6.21 | 113.7 | 11.41 |
| flash | 128 | 253.6 | 12.21 | 113.8 | 22.59 |
| flash | 192 | 250.9 | 18.24 | 112.9 | 33.79 |
| flash | 256 | 246.5 | 24.21 | OOM | OOM |

At batch 128, Flash reduces allocated memory by 13.9% for CLM and 16.8% for joint,
while improving throughput by 33.7% and 34.5%, respectively. It lets joint batch
192 complete where eager batch 192 fails. That successful Flash case reserves
42.65 GiB, close to the 44.39 GiB device capacity reported by the allocator; joint
batch 256 still fails. CLM batch 256 fits on both backends. These are observed
limits for this allocator, objective, and batch grid, not universal capacity limits.

**Use Flash with physical batch 64 for the next quality pilot.** In the extended
sweep, batch 128 gives effectively the same throughput while doubling allocated
memory. The initial sweep was about 4-5% faster at 128 than 64, so the exact
plateau needs longer timing. Batch 192/256 does not improve observed speed.
Against eager batch 16 in the extended run, Flash batch 64 improves CLM throughput
by 50.1% and joint by 56.5%. At the same batch 64, kernel improvements are 32.5%
and 31.9%. The batch benefit and the kernel benefit should not be conflated.

Both sweeps confirmed native Flash forward/backward dispatch, finite gradients,
and zero prefix change after altering a future token. Maximum eager/Flash BF16
absolute differences were 0.005343 for hidden states, 0.0078125 for LM logits,
and 0.0004883 for RTD logits, within the recorded tolerances. Floating-point
rounding and different dropout RNG consumption mean training trajectories need
not be bitwise equal. CPU formatting/linting and all **125 tests** pass.

- [Initial sixteen-case raw result](results/modal-l40s-tinystories15m-20261001-attention.json)
- [Extended sweep, including OOM errors and reserved memory](results/modal-l40s-tinystories15m-20261001-attention-capacity.json)
- [Source commits, file hashes, optimizer settings, and app status](results/attention-provenance-20261001.json)
- [Initial Modal app](https://modal.com/apps/gaon12/main/ap-0TRGg8GdfKumDZJOndI2iV)
- [Extended Modal app](https://modal.com/apps/gaon12/main/ap-I0aWPbc8JenOZQu6aWmUn0)

Measured GPU benchmark functions took 69.88 and 113.18 seconds. Their combined
GPU-only estimate is $0.0992 at the recorded $0.000542/second rate; this excludes
startup/scaledown, pre-timer data loading, CPU, host memory, storage, and egress.
Both apps were stopped with zero remaining tasks when checked at 16:11:25 KST.
These benchmark runs produce no quality checkpoint or validation PPL.

## Conditional longer-training estimate

At the extended run's Flash batch-64 throughput, the following arithmetic estimates
only optimizer training. They do not claim sustained performance or target quality.

| Input tokens per condition | CLM hours / GPU estimate | Joint hours / GPU estimate | Both training hours |
| --- | --- | --- | --- |
| 100M | 0.110 / $0.21 | 0.244 / $0.48 | 0.354 |
| 1B | 1.096 / $2.14 | 2.442 / $4.77 | 3.539 |
| 3B | 3.289 / $6.42 | 7.327 / $14.30 | 10.616 |

Formula: tokens divided by measured tokens/s, with cost multiplied by the recorded
GPU-second rate. Validation, saving, data preparation, startup, and all other
charges are extra. The same longer-training infrastructure described in TARGET
is still required: LR warmup/cosine, optimizer/RNG resume, and data shards for
larger unique-data budgets. A 1B input budget does not promise reference quality.

## Reproduction

```powershell
.venv/Scripts/python -m modal run attention_app.py::attention_main --run-id attention-unique-run
```

Use a new output ID. The CPU preparation function reuses the pinned dataset cache
before allocating the GPU. The function timeout is 600 seconds, with an internal
480-second budget check between cases. Results are saved to the Modal Volume and
`results/<run-id>.json`; benchmark weights are deliberately not saved as trained
candidates. The profiler confirms kernel dispatch, not target-quality training.

An example local CUDA pilot using the packed target dataset is:

```powershell
.venv/Scripts/python -m deletcra run --preset story15m --mode joint --data-dir data/tinystories-benchmark --sequence-length 256 --attention-backend flash --device cuda --precision bf16 --batch-size 64 --steps 300 --probe-steps 0 --share-embeddings --output-dir runs/flash-joint-unique
```

For CPU inspection of saved Flash weights, explicitly load with
`CausalElectra.load(path, attention_backend="sdpa")`. This switches the execution
kernel while preserving the weights. Attention-weight output, head masks,
cross-attention, and caching are not supported by this adapter.

## Training interpretation

Larger physical batches amortize small-model launch and optimizer overhead.
Flash also avoids storing dense attention probabilities. The 32,000-way
vocabulary projection and cross-entropy still consume memory; attention is only
one part of the training footprint at context 256. Savings should be measured
for the complete objective, especially the generator and two main joint passes.

At fixed input tokens, batch 128 takes eight times fewer optimizer steps than
batch 16; the recommended batch 64 takes four times fewer. Faster token throughput
therefore does not establish equal convergence
or story quality. Learning rate and warmup/cosine schedule need validation at the
chosen batch. Gradient accumulation changes effective batch, but does not replace
the throughput benefit of a larger physical batch in this measurement.

Keep the 15M scale for the first quality experiment. The
[TinyStories paper](https://arxiv.org/abs/2305.07759) demonstrates coherent stories
with models below 10M parameters, supporting this restricted target. DELECTRA has
not matched the reference: the current common-protocol pilot PPL is about 42/47
for CLM/joint versus 3.94 for the reference. Measure a properly scheduled learning
curve before concluding that parameters are the limiting factor. See
[TARGET.md](TARGET.md) for remaining longer-training work and comparison limits.

[GPU_COMPARISON.md](GPU_COMPARISON.md) extends these short L40S measurements
with repeated fixed-batch runs on L40S, A100 80GB, RTX PRO 6000 and H100.
Use those longer measurements for the current conditional compute estimate.
