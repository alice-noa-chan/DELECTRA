# Colab TPU versus Runpod GPU

Evidence reviewed on October 2, 2026. No Colab runtime was allocated and no TPU
throughput was measured for this comparison. The current trainer supports CPU
and CUDA; CPU verification does not establish TPU compatibility.

## Decision

Try a short free Colab TPU correctness and throughput experiment before paying
for another GPU run. Free accelerator access can reduce cash spending, but it
does not establish a completion date. Keep Runpod as the measured fallback.
Full TinyStories, Base, and IT publication still requires completed training and
quality evaluation. Base corpus cleaning remains an unresolved quality gate.

| Item | Runpod RTX 3090 | Free Colab TPU |
| --- | --- | --- |
| DELECTRA evidence | Three batch-256 repetitions measured | 15M BF16 joint update and same-device resume verified; sustained throughput unmeasured |
| Accelerator charge | Observed $0.22/hour on October 1 | $0 only when allocated under free-tier access |
| Current training kernels | BF16, PyTorch CUDA Flash Attention, Liger | XLA BF16, eager attention, dense fixed-shape vocabulary losses/proposals |
| Base training-step estimate | 32.46 hours for 16.4B nominal input positions | Depends on measured sustainable throughput |
| Base charge estimate | $7.27 including the measured container rate | Free accelerator; persistence or paid compute can add costs |
| Continuity | Provider availability and interruption risks remain | Variable quota; free sessions at most 12 hours, potentially shorter |
| Reproducible restart | CPU and Colab GPU FP32/FP16 restart verified | Tied 15M BF16 whole/resumed weights and Adam state match; explicit T4 FP16 handoff passed |

Runpod figures derive from [the archived measurement](results/runpod-3090-20261001.json)
and [its protocol](BUDGET_GPU.md), not a current price guarantee. Batch 256 reached
139,815.9246 next-token targets/second on the 15,041,505-parameter joint self-RTD
model at context 256. This was a short cached-data benchmark, not the complete
production data loader or a convergence measurement. The historical offer added
$0.004/hour for a 30GB container. Setup, validation, test, persistent storage and
IT are excluded. The earlier conservative $7.30 used 16.4B prediction targets
instead of nominal input positions; neither figure covers all three releases.
[Runpod storage pricing](https://docs.runpod.io/pods/pricing) lists separate
container, volume and network-volume charges.

## CLI and free-tier constraints

The [official Colab CLI](https://github.com/googlecolab/google-colab-cli) supports
TPU requests, code execution, file transfer and session inspection. Its README
currently lists v5e1 and v6e1 and supports Linux/macOS, excluding Windows. This
machine has an Ubuntu WSL distribution. [COLAB.md](COLAB.md) records subsequent
CLI installation, authentication and actual free T4/v5e1 checks. The later
single-device XLA implementation is documented in
[TRAINING_RUNTIME.md](TRAINING_RUNTIME.md). No Runpod allocation is authorized
for the current Colab work; its numbers remain historical evidence.
An accepted hardware option does not promise free account entitlement or stock.

The [Colab FAQ](https://research.google.com/colaboratory/faq.html) says free
resources and hardware types vary, and free sessions can run for at most 12
hours depending on availability and usage. It also restricts SSH/remote desktop
on free runtimes without a positive compute-unit balance. CLI functionality
does not override those policies: prefer notebook execution for free testing
and do not assume its SSH workflow is eligible. No account-specific entitlement
was checked for this initial comparison; subsequent checks are in COLAB.md.

Google lists per-chip HBM of [16GB for v5e](https://cloud.google.com/tpu/docs/v5e)
and [32GB for v6e](https://cloud.google.com/tpu/docs/v6e). Those are hardware
specifications, not confirmed Colab allocations, host RAM or predicted DELECTRA
speed. A TPU request is not an allocation of an entire multi-chip TPU Pod.

## Implementation and remaining optimization

Preserve ELECTRA embeddings, transformer blocks, learned absolute positions and
the discriminator head. Preserve shifted learned proposals, detached sampling,
actual-change RTD labels, and joint loss weights. Hardware migration must not
silently turn the main experiment into CLM-only training.

1. An optional XLA runtime, BF16 autocast, synchronization and portable RNG/
   checkpoint handling are implemented. Logging still reads device scalars and
   needs profiling before throughput claims. The
   [PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html)
   describes lazy execution and step synchronization; the
   [AMP guide](https://docs.pytorch.org/xla/master/perf/amp.html) documents TPU BF16.
2. Proposal sampling, CLM/RTD loss and response-only SFT now use fixed-shape
   dense masked computations on XLA; CPU equivalence tests pass.
   Python conditions on device tensors also need review. The
   [XLA recompilation guide](https://docs.pytorch.org/xla/master/perf/recompilation.html)
   explains dynamic outputs and host synchronization. This is a local-code
   performance concern; the actual short TPU check records compilation and host
   scalar reads separately from graph execution.
3. Eager attention and dense vocabulary loss execute on XLA. Existing forced CUDA
   Flash Attention, Liger loss and fused CUDA AdamW cannot simply be enabled on
   TPU. Validate the new kernels against the reference losses and gradients.
   A dense BF16 vocabulary tensor at batch 256 has roughly 4.18GB of elements
   for [256,255,32000], before gradients, FP32 intermediates and activations;
   chunking may be needed even though the model has only 15M parameters.
4. Prefetch fixed-shape batches from local VM storage before sustained training.
   Partial batches are preserved through BOS-only padding and loss masks.
   Prepared corpora occupy roughly 15.4GB before ancillary files/checkpoints;
   check durable storage capacity first. Copy data to VM storage rather than
   randomly reading mmap blocks through Drive, consistent with the FAQ's I/O
   guidance. Save resumable checkpoints outside the ephemeral runtime.

## Time sensitivity, not TPU predictions

For fully packed Base blocks, next-token targets equal nominal positions times
255/256. Thus the accepted 16.4B-input plan has 16,335,937,500 targets. Active
training hours equal that count divided by targets/second and by 3,600.

| Hypothetical measured targets/second | Base active training hours | Ideal sessions if each provided 12 usable hours |
| --- | --- | --- |
| 50,000 | 90.76 | 8 |
| 100,000 | 45.38 | 4 |
| 200,000 | 22.69 | 2 |

These rates are arbitrary sensitivity points, not TPU benchmarks. One 12-hour
session would require about 378,147 targets/second before overhead. Waiting for
quota, transferring data, compiling, evaluation and saving/reloading checkpoints
increase elapsed time and session count. Calendar completion cannot be inferred
from active compute hours alone.

## Fair pilot protocol

Use the same pinned benchmark corpus, model configuration, context, dropout and
joint objective as the Runpod archive. Verify causal prefix isolation, t-1 to t
proposal alignment, unchanged-sample labels, gradient paths, finite BF16 losses
and restart correctness before interpreting speed. Do not require GPU/TPU
bitwise equality across different kernels and RNG implementations.

Warm up until compilation stabilizes, then measure at least three synchronized
trials. Record compilation time separately from sustained targets/second;
inspect XLA recompilations and CPU fallbacks. Compare equivalent effective
batches before tuning physical batch size, since a larger update batch changes
optimization. Follow with the actual production loader and a durable-checkpoint
round trip. Report the allocated TPU type/chip count, environment, memory,
source commit, token counts and account eligibility without exposing credentials.
Only those measurements can support a full TinyStories/Base/IT time estimate.
