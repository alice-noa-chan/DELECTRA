# Colab CLI setup and free compute handoff

## Installed host

On October 2, 2026 the official `google-colab-cli==0.7.4` was installed with
`uv tool install` in Ubuntu 26.04 on WSL2. Its isolated tool environment uses
Python 3.12.13, separate from Windows DELECTRA and system Python 3.14.4. Version,
session/usage/execute help and a 52-package dependency check passed.

The [official CLI](https://github.com/googlecolab/google-colab-cli) supports
Linux/macOS and TPU v5e1/v6e1 requests. The installed help defaults to `oauth2`,
even though the repository README inspected during the first comparison listed
`adc`. Use installed command help for this pinned version.

Authenticate from an interactive local PowerShell terminal:

```powershell
wsl -d Ubuntu-26.04 -- bash -lc 'colab usage'
```

Follow the printed Google URL and paste its authorization code into that
terminal. Never put authorization codes, token files or CLI session metadata
in Git or research logs. Login succeeded: the account initially had 0.00 CU,
zero active assignments and zero usage rate. A free T4 allocation subsequently
succeeded; a v5e1 request timed out after 120 seconds with no observed assignment.
That timeout does not prove permanent TPU ineligibility.

## Bounded verification

Use `colab sessions` and `colab usage` to inspect actual allocation and account
status. Avoid allocating another runtime while a previous request has an
unknown outcome. A displayed compute-unit hourly rate is not a dollar invoice;
record balance and rate separately. No paid plan or compute units were bought.

`src/deletcra/runtime_probe.py` can run before DELECTRA is installed. It reports
hardware and versions and checks matrix forward/backward results. A requested
accelerator must really be present; it cannot silently pass on CPU. The CUDA
probe checks FP16 and tests BF16 only when supported natively, excluding CUDA
emulation. It does not establish model compatibility or training throughput.

From WSL in `/mnt/d/models/DELETCRA`, for an existing verification session:

```bash
colab exec -s delectra-gpu-probe --timeout 180 \
  --env DELECTRA_PROBE_DEVICE=cuda \
  --env DELECTRA_PROBE_OUTPUT=/content/delectra-runtime-probe.json \
  -f src/deletcra/runtime_probe.py
```

For a TPU session, use `DELECTRA_PROBE_DEVICE=xla` and its session name. The VM
needs a compatible Torch/XLA pair; do not replace the runtime's packages blindly.
The current DELECTRA trainer itself still excludes XLA.

After installing the project and compatible Transformers, run a bounded check:

```bash
python -m deletcra.verify_training --device cuda --output /content/trainer-check
```

This verifies the real 15,041,505-parameter model with four synthetic FP32 joint
RTD/CLM updates, comparing uninterrupted training with pause/resume. It checks
weights, Adam moments, token counts and the data cursor within explicit numerical
tolerances. It is not a real-data benchmark, full training or quality evaluation.
Free T4 hardware lacks native BF16. FP16 matrix support does not enable FP16
production training: the current trainer supports FP32 and CUDA BF16 only.
FP16 training needs loss scaling for both backward paths and checkpointed scaler
state before it can replace FP32 safely.

Download reports and check their hashes before releasing the temporary runtime.
`colab stop -s <name>` deletes runtime-local files; retained research evidence
must be exported first. Check `colab sessions` and `colab usage` afterward.

## TPU, then Colab GPU, then Runpod GPU

Prefer whichever free accelerator is available and verified for the objective;
the order is a preference, not a prerequisite to making progress. Preserve
RTD and the ELECTRA backbone throughout. Use Runpod only for the unfinished
training budget after free access, with a separately selected spending cap.
No automatic paid fallback is configured.

There are three distinct checkpoint situations:

1. Colab GPU to Runpod GPU with matching Torch/configuration: copy the complete
   run, corpus manifest and checkpoint, verify SHA-256, and resume optimizer,
   sampler and scheduler state. Weights alone start a different experiment.
   Actual transfer between providers remains to be verified.
2. Different GPU capability or backend: changing BF16/FP32, attention, loss
   kernel, physical batch or fused optimizer currently changes the saved spec,
   so strict resume refuses it. A future explicit migration must keep the
   research recipe invariant, define allowed execution changes, and record
   source/target environment and consumed-token cursor.
3. TPU to GPU: additionally translate XLA optimizer storage and device RNG
   handling. Restore the same cumulative input budget and LR schedule, not a
   new warmup. Different RNG implementations do not provide a bitwise identical
   continuation; record the boundary and verify finite gradients and comparable
   validation metrics. This route is not implemented or verified yet.

TPU and GPU requests do not promise independent free quotas. The
[Colab FAQ](https://research.google.com/colaboratory/faq.html) describes dynamic
limits, potentially shorter sessions up to 12 hours, and free-tier restrictions
on SSH/remote desktop. CLI use does not remove these limits. Use notebook/kernel
execution, save durable checkpoints periodically, and wait when free access is
exhausted. Account cycling or artificial idle prevention is not part of this
plan. [TPU_COMPARISON.md](TPU_COMPARISON.md) preserves the measured Runpod baseline
and explicitly hypothetical TPU timing scenarios.
