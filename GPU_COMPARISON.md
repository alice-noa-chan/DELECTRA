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

Formatting/linting and 128 CPU tests passed before the cloud run, including tests
that reject partial or changed-batch comparisons, incorrect GPU identity, and
ranking by raw speed instead of cost. Cloud evidence verifies CUDA execution.
