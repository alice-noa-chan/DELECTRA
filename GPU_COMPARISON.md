# Fixed-batch GPU cost comparison

This experiment answers whether a faster, more expensive GPU reduces the cost
of the accepted 16.4B-input-token Base pretraining plan. It benchmarks existing
causal ELECTRA training code; it does not start full-corpus training or publish
model weights.

## Protocol

All requests use one exact GPU, the same image, and the same packed TinyStories
cache. Requested types are L40S, A100-80GB, RTX-PRO-6000 and H100!. The H100!
request disables Modal's possible H200 upgrade, and actual device names and
memory capacity are validated before measuring.
[Modal GPU documentation](https://modal.com/docs/guide/gpu).

The model remains 15,041,505-parameter causal ELECTRA. Joint RTD+CLM shares
generator token/position embeddings and optimizes 15,239,402 unique parameters.
Only the GPU changes: context 256, physical batch 64, CUDA BF16 autocast with
FP32 parameters, forced native FlashAttention, dropout 0.1, seed 7, AdamW
learning rate 0.0003, weight decay 0.01, clipping 1.0, and the same 1/5/1
generator/RTD/main-CLM loss weights.

For each GPU and objective, run **three independent fresh-model cases**, each
with twenty warmup optimizer steps and two hundred measured optimizer steps.
Each measured case processes 3,264,000 input-token targets, counting 255 per
256-token block. Mode order alternates between repetitions. Batches and model
initialization use the same seeds; hardware kernels can still differ numerically.
Each GPU processes 19,584,000 measured input targets across its six cases.

Timing includes batch transfer, finite-loss checks, backward, gradient clipping
and AdamW. Validation, saving, and a separate operator-profiler step are excluded.
Aggregate throughput is total measured tokens divided by total optimizer time,
not the fastest repetition. Sample standard deviation across three throughputs
is reported as a percentage; it is not a statistical confidence interval.

Actual CUDA correctness checks compare Flash/eager outputs and require zero
prefix change after replacing a future token. Every measured step requires finite
loss/gradients. The profiler requires six native Flash forward/backward calls
for CLM and fourteen for joint, including the generator and both main passes.
Unsupported execution is recorded as a failure, never silently replaced with
another attention backend.

## Pricing and interpretation

Rates were checked on 2026-10-01 at [Modal pricing](https://modal.com/pricing).

| Request | GPU USD/second | GPU USD/hour | Required speedup over L40S for lower GPU cost |
| --- | --- | --- | --- |
| L40S | 0.000542 | 1.9512 | 1.000x |
| A100-80GB | 0.000694 | 2.4984 | 1.280x |
| RTX-PRO-6000 | 0.000842 | 3.0312 | 1.554x |
| H100! | 0.001097 | 3.9492 | 2.024x |

GPU USD/million tokens equals the GPU second rate times one million divided by
aggregate tokens/s. Base hours equal 16.4B divided by tokens/s and 3600; Base GPU
cost equals the GPU second rate times those training seconds. Rank Base economics
using **joint**, with CLM retained as an internal control.

All containers request two CPU cores and 8GiB host memory. Requested host-resource
rates add USD 0.00004396/second. The report also computes a requested-compute
estimate, but Modal bills the greater of requested and actual resource use;
these projections are not metered invoices. Region premiums, startup/scaledown,
data preparation, evaluation, checkpoints, storage, egress and credits are
excluded. GPU function cost estimates exclude application-load time before the
function starts and persistence work after its recorded timer.

The comparison holds physical batch fixed so changing optimizer updates per token
cannot explain speed differences. It does not prove equal final quality across
hardware. Larger batches, context changes, a new sharded-corpus loader, or CPU
bottlenecks require another profile. Multi-GPU training is outside this experiment.

## Reproduction and bounds

```powershell
.venv/Scripts/python -m modal run gpu_app.py::compare_main --run-id gpu-cost-unique-run
```

Outputs must be new. Four independent GPU functions share read-only cached data
and persist results in distinct Volume folders. Each permits one container, zero
retries, a 600-second timeout and a 450-second internal budget check between cases.
Local JSON saves each completed worker independently before writing the aggregate
cost ranking. The source commit and pinned dataset metadata accompany each result.

Formatting/linting and 128 CPU tests passed before the first cloud run, including tests
that reject partial or changed-batch comparisons, incorrect GPU identity, and
ranking by raw speed instead of cost. Cloud evidence verifies CUDA execution.

## Four-GPU measurements: 2026-10-01

All four requests completed all six cases on their validated physical devices.
Every case passed finite-gradient checks and native Flash forward/backward
profiling. All four eager/Flash checks passed, with exactly zero causal prefix
difference. The shared runtime was PyTorch 2.8.0+cu128, CUDA 12.8, and
Transformers 4.57.6. Reported memory is peak PyTorch allocated training memory.

| GPU | Joint input tokens/s | Repetition sample SD | Joint peak GiB | 16.4B training hours | 16.4B GPU USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| L40S | 108,198 | 0.05% | 11.41 | 42.10 | 82.15 |
| A100 SXM4 80GB | 130,619 | 1.30% | 11.41 | 34.88 | 87.14 |
| RTX PRO 6000 Blackwell Server Edition | 181,839 | 3.20% | 11.41 | 25.05 | 75.94 |
| H100 80GB HBM3 | 232,518 | 1.45% | 11.45 | 19.59 | 77.37 |

H100 was 2.15 times as fast as L40S for joint training. A100's 1.21-times speedup
did not offset its price: the projected GPU cost increased about 6.1%. RTX and
H100 both reduced projected GPU cost relative to L40S. RTX's approximately 1.9%
cost advantage over H100 was small compared with the observed repeat variation,
so it warranted longer confirmation rather than a definitive cheapest-GPU claim.

CLM-only control throughputs were 241,452, 287,445, 420,429 and 530,045 tokens/s
for L40S, A100, RTX and H100, respectively. Their projected GPU costs were
USD 36.81, 39.60, 32.84 and 33.94 for 16.4B tokens. These are different objectives
with less compute per input token and do not replace the joint Base estimate.

The four measured function durations total an estimated USD 0.3385 in GPU
charges under the exclusions above. Preserve the complete measurements in the
[four-GPU aggregate](results/modal-gpu-story15m-20261001-comparison.json).
The source commit is `7f4cf651c464cda621ec609159dc5d5528361337`; the
[Modal app](https://modal.com/apps/gaon12/main/ap-pj7RC9g6DV3lULP5IBi8aY)
was verified stopped with zero tasks at 2026-10-01 16:48:25 +09:00.

## Longer joint confirmation

The separate confirmation compares RTX PRO 6000 and exact H100 only. Each GPU
runs three fresh joint cases with twenty warmup steps and **1,000 measured steps**
per case: 48,960,000 measured input targets per GPU. All other settings and
correctness checks remain the same. The shorter and longer phases are kept in
separate reports and are not pooled into one ranking.

```powershell
.venv/Scripts/python -m modal run gpu_app.py::confirm_main --run-id gpu-cost-confirm-unique-run
```

The confirmation implementation passed formatting/linting and all 129 CPU
tests, including rejection of short cases supplied as long confirmation cases.

| GPU | Joint input tokens/s | Repetition sample SD | 16.4B training hours | 16.4B GPU USD | 16.4B requested compute USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 Blackwell Server Edition | 207,785 | 0.17% | 21.92 | 66.46 | 69.93 |
| H100 80GB HBM3 | 233,509 | 3.59% | 19.51 | 77.05 | 80.13 |

RTX reduced projected GPU cost by **13.7%** relative to H100 in this phase;
H100 reduced optimizer time by **11.0%**, or about 2.42 hours at 16.4B inputs.
RTX's cost advantage held even when comparing its slowest repetition with H100's
fastest. This supports **RTX PRO 6000 for cost** and **H100 for elapsed time**
for this model and batch, rather than choosing the GPU by its hourly price alone.
Requested CPU/host-memory estimates retain the same ranking.

RTX throughput was 14.3% higher than in the first phase, while H100 was similar.
These phases use different cloud allocations and measurement lengths; the
experiment cannot isolate the cause of that change. Within-phase repetition SD
does not capture all allocation-to-allocation variation. For budgeting, preserve
the observed RTX range: approximately **22-25 training hours and USD 66-76
GPU-only**. A longer recent-corpus pilot must confirm throughput after replacing
the in-memory cache with the production loader. This is not a convergence or
final model-quality result.

All six long cases passed finite-gradient and native Flash forward/backward
checks. Both GPUs again had exactly zero causal prefix difference. Peak allocated
joint memory remained about 11.41GiB on RTX and 11.45GiB on H100. Neither trial
trained the complete TinyStories corpus or the planned recent-data Base.

The source commit is `c9b451fea7d04c4ee0eba2f7414f82fd6c1ce777`.
See the [confirmation aggregate](results/modal-gpu-story15m-20261001-confirmation.json)
and [Modal app](https://modal.com/apps/gaon12/main/ap-p7iG5bTK6zIBafZjMigVok).
Both apps were verified stopped with zero tasks after confirmation. The
[provenance manifest](results/gpu-comparison-provenance-20261001.json) records the
status-check timestamp, source hashes, and SHA-256 hashes of all eight result
files using UTF-8 with LF line endings for cross-platform Git verification.
Recomputing both aggregates reproduced them exactly; all 30 cases account
for 176,256,000 measured input targets. The two phases total approximately
**USD 0.788 GPU-only estimated function charges**, subject to the exclusions above.

## Subsequent integrated-model experiment

[OPTIMIZATION.md](OPTIMIZATION.md) records a later RTX PRO 6000 comparison of
separate versus self-replacement joint training, Liger loss, backward execution,
AdamW fusion and batches 64/256/512. Its current separate baseline is retained
within that run; these historical four-GPU measurements remain unchanged.
The measured self batch-256 configuration substantially reduces token cost,
but changes the proposal distribution and update count. Original separate-model
costs above are not predictions for the integrated variant on other GPUs.
Full training and quality equivalence remain unverified.

[BUDGET_GPU.md](BUDGET_GPU.md) adds a completed Community Runpod RTX 3090
measurement of the integrated configuration, at $0.22/GPU-hour rather than the
public Secure Cloud offer. Its batch-256 projection is 32.58 hours / $7.17
GPU-only for 16.4B targets. This is substantially slower and cheaper than the
measured Modal RTX PRO 6000 integrated configuration. Preserve the distinction
between different objectives, allocations, prices and production-loader costs.
