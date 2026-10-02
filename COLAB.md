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
succeeded. The first v5e1 request timed out after 120 seconds, and immediate
inventory showed no assignment. It subsequently appeared as a delayed TPU
allocation. Recovering its exact endpoint into local CLI session state allowed
the hardware check without issuing another allocation or exposing credentials.

## Initial hardware and FP32 verification

| Check | Result | Scope |
| --- | --- | --- |
| Colab T4 | FP32 and FP16 matrix gradients passed; native BF16 unavailable | Actual CUDA hardware |
| T4 DELECTRA | 15,041,505 parameters; four synthetic joint RTD/CLM FP32 steps; interrupted resume matched weights with zero maximum error and matched Adam moments/data cursor | 1,024 input positions; no language-quality or throughput claim |
| Colab v5e1 | One XLA TPU device; FP32 and native BF16 matrix gradients passed | Hardware only; no DELECTRA training or handoff verified |

The T4 image used Python 3.13.15 and Torch 2.11.0+cu130. Transformers was pinned
to 4.57.6 before the model check. The TPU image already provided Torch 2.9.0+cpu,
Torch/XLA 2.9.0 and libtpu 0.0.21.1; those packages were not replaced.

The T4 report and resumable synthetic checkpoint were downloaded and verified
against remote SHA-256 and byte counts before an explicit stop. The checkpoint
is retained privately at `runs/colab-t4-verification-20261002/latest.pt`. The TPU
session disappeared before its report could be downloaded or the project
installed. Its successful hardware result is preserved from CLI stdout, without
claiming a verified downloaded artifact. The cause of termination is unknown.
Final server inventory showed zero assignments, zero CU usage rate and 0.00 CU
balance. No paid hardware was allocated or compute units purchased.

[The verification record](results/colab-free-access-verification-20261002.json)
links individual reports, provenance, export hashes and limitations. CLI host
package versions are in [the WSL environment record](results/colab-cli-wsl-20261002.json).

## Later mixed-precision implementation checks

CUDA FP16 whole-versus-resumed training passed on a free T4: four applied joint
updates, no skipped updates, identical model weights (maximum error 0), matching
Adam moments/data cursor and restored GradScaler state. This used 1,024 synthetic
input positions, not TinyStories training or a language-quality measurement.
[FP16 result](results/colab-t4-fp16-verification-20261002.json).

The first XLA model updates exposed two real portability issues: synchronization
advanced the device seed, and conversion split the tied vocabulary table into
independent input/output parameters. The latter produced 24,257,505 parameters
and 111 Adam entries, so it is an out-of-spec diagnostic, not evidence for the
intended tied 15M model. Its original reports and checkpoints remain archived.
The attempted CUDA handoff correctly failed with an optimizer group mismatch;
no trained moments or divergent weights were silently merged. The implementation
now reties weights after conversion, rejects divergent checkpoints, and preserves
the next-step seed during checkpoint synchronization. See
[TRAINING_RUNTIME.md](TRAINING_RUNTIME.md) for the corrected execution contract.

The corrected tied 15,041,505-parameter model passed whole-versus-resumed BF16
joint training on v5e1: matching Adam state/data cursor, zero maximum weight error,
110 optimizer entries and finite nonzero RTD gradients. Its two-step run consumed
512 input positions; compilation and scalar logging are included in the recorded
XLA metrics. This is not a sustained throughput measurement.
[Corrected TPU result](results/colab-tied-xla-verification-20261002.json).

That TPU checkpoint was exported, hash-verified on the T4 destination, and
explicitly migrated to CUDA FP16/SDPA/fused AdamW. Weights, Adam moments/step and
the consumed-data cursor were exactly preserved before the next applied update.
The recorded boundary reseeds device/proposal RNG and starts an FP16 scaler;
cross-device bitwise continuation is not claimed.
[TPU-to-GPU result](results/colab-tpu-gpu-handoff-20261002.json).

A separate real TinyStories mmap pilot used 256 sampled training blocks and 16
validation blocks from the audited prepared corpus, with upstream identity,
manifest hash and source indices retained. Four batch-16 FP16 joint updates
consumed 16,384 input positions and 16,320 prediction targets without skipped
updates. Peak CUDA allocated memory was 2,118,268,928 bytes (1.97 GiB); this is
not total process VRAM or proof that all larger batches fit. The early validation
NLL is 10.0622; it is not independent-test or finished-model quality evidence.
[Real-data pilot](results/colab-real-stories-verification-20261002.json).

Full TinyStories/Base/IT training and publication remain outstanding. These
checks preserve ELECTRA blocks, tied vocabulary embeddings and the RTD head;
RTD-only, CLM control and joint objectives remain supported. GPU execution above
uses joint self proposals; separate-generator and SFT XLA behavior has CPU tests
but is not independently verified on TPU hardware. Warmed production-loader
profiling, larger-batch tuning, and XLA vocabulary/logging optimization are still
needed before making total-training-time estimates.

The [mixed-precision verification record](results/colab-training-runtime-20261002.json)
links reports, source/wheel provenance, protocols and hash-verified private
checkpoint archives. No pilot checkpoint or sample data is published to Hugging
Face or committed to Git.
Inspect the report's `status` field: this CLI can exit successfully while a
remote Python cell raised an exception. The archived check reports, rather than
the process exit code alone, establish the results above.
All four runtimes created for these checks were explicitly stopped after verified
export. Final assignment inventory was empty and the CU balance remained 0.00.
The separate consumption-info snapshot still reported one assignment and
0.8025 CU/hour; its disagreement with inventory is preserved without assuming
the cause or treating CU/hour as a dollar invoice.
[Cleanup snapshots](results/colab-training-cleanup-20261002.json).

## Bounded verification

Use `colab sessions` and `colab usage` to inspect actual allocation and account
status. After a timeout, recheck server inventory and recover any observed
assignment before retrying the same hardware request. An initially empty list
does not rule out a delayed assignment. A displayed CU hourly rate is not a dollar invoice;
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
The production trainer now supports single-device XLA BF16. Its fixed-shape
objectives and explicit checkpoint migration are described in
[TRAINING_RUNTIME.md](TRAINING_RUNTIME.md).

After installing the project and compatible Transformers, run a bounded check:

```bash
python -m deletcra.verify_training --device cuda --output /content/trainer-check
```

This verifies the real 15,041,505-parameter model with four synthetic FP32 joint
RTD/CLM updates, comparing uninterrupted training with pause/resume. It checks
weights, Adam moments, token counts and the data cursor within explicit numerical
tolerances. It is not a real-data benchmark, full training or quality evaluation.
Free T4 hardware lacks native BF16. CUDA FP16 production training now scales both
backward paths, clips only after unscaling, and checkpoints the GradScaler. Add
`--precision fp16` to the verification command to check that execution path.

Download reports and check their hashes before releasing the temporary runtime.
`colab stop -s <name>` deletes runtime-local files; retained research evidence
must be exported first. Check `colab sessions` and `colab usage` afterward.

## TPU, then free Colab GPU

Prefer whichever free accelerator is available and verified for the objective;
the order is a preference, not a prerequisite to making progress. Preserve
RTD and the ELECTRA backbone throughout. The current user instruction excludes
Runpod. No paid fallback is configured; stop or wait when free quota is exhausted.

There are three distinct checkpoint situations:

1. Colab GPU to Runpod GPU with matching Torch/configuration: copy the complete
   run, corpus manifest and checkpoint, verify SHA-256, and resume optimizer,
   sampler and scheduler state. Weights alone start a different experiment.
   Actual transfer between providers remains to be verified.
2. Different GPU capability or backend: strict resume refuses specification
   changes. Explicit `--resume --migrate` allows device, precision, attention/loss
   kernel, fused optimizer, host threads and wall allowance changes while keeping
   the research recipe, physical batch, data, input budget and schedule invariant.
   Record the source/target environment and consumed-token cursor.
3. TPU to GPU: additionally translate XLA optimizer storage and device RNG
   handling. Restore the same cumulative input budget and LR schedule, not a
   new warmup. Different RNG implementations do not provide a bitwise identical
   continuation; record the boundary and verify finite gradients and comparable
   validation metrics. The implementation uses portable CPU tensor storage and
   records a deliberate device/proposal RNG reseed at the migration boundary.

TPU and GPU requests do not promise independent free quotas. The
[Colab FAQ](https://research.google.com/colaboratory/faq.html) describes dynamic
limits, potentially shorter sessions up to 12 hours, and free-tier restrictions
on SSH/remote desktop. CLI use does not remove these limits. Use notebook/kernel
execution, save durable checkpoints periodically, and wait when free access is
exhausted. Account cycling or artificial idle prevention is not part of this
plan. [TPU_COMPARISON.md](TPU_COMPARISON.md) preserves the measured Runpod baseline
and explicitly hypothetical TPU timing scenarios.
