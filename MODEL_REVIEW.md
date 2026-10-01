# DELECTRA model review

This is a research model, not yet evidence that causal RTD improves generation.
Its ELECTRA identity is the original backbone and discriminator head, learned
replacement proposals, and detection at every eligible content position.
Causal attention and next-token learning are documented departures.

## The uncomfortable findings

**15M is a size, not a quality result.** Of 15,041,505 parameters, 9,216,000
(61.3%) are the 32,000-by-288 token table. The preset has six backbone layers.
Original ELECTRA-small used 128-wide embeddings and a twelve-layer, 256-wide
backbone. Narrower embeddings could free capacity for depth while preserving
ELECTRA's embedding projection. That is an architecture ablation, not an
execution optimization; preserve existing preset and checkpoint shapes.
[Original paper](https://cs.stanford.edu/~kevclark/resources/electra.pdf).

**The pilot is far from the target.** Our shared 99,960-target validation protocol
gave the archived 300-step joint pilot perplexity 47.27, CLM control 41.87 and
reference approximately 3.94. Samples repeat phrases and lose narrative
consistency. These undertrained pilots cannot rank final architectures or prove
RTD hurts, but release quality has not been achieved. Reference training overlap
is unknown: this is not an independent reference test. See [RESEARCH.md](RESEARCH.md)
and [joint evidence](results/tinystories15m-joint-pilot-benchmark-20261001.json).

**Integration still needs two main passes.** Clean CLM provides proposals; RTD
must then see genuinely corrupted inputs. Both heads on clean hidden states
would not train detection of corrupted tokens. The embedding-shared separate
generator adds only 197,897 unique parameters. Its runtime, vocabulary projection
and activation graph are the main targets. Self replacement changes proposals,
so it is not algorithmically equivalent to the smaller generator.

**A good generator can starve RTD of positives.** An unchanged sampled token must
keep label zero. Actual replacements can fall below the 15% selection rate and
ordinary accuracy becomes misleading. Monitor actual replacement rate, majority
baseline, balanced accuracy, AUROC, AP, recall and held-out CLM loss together.
Do not force different samples or manufacture positive labels. Temperature and
RTD weight require matched quality experiments.

**Causal RTD is a different task.** A position sees its current token and left
context, but lacks ELECTRA's right context. Proposals use original prefixes;
RTD sees corrupted prefixes. Parallel clean-prefix sampling is deliberate, not
recursive sampling from a corrupted story. Dense RTD survives, but original
bidirectional sample-efficiency findings do not automatically transfer. The
RTD-only main LM head is untrained and is not a story generator.

**The production training protocol is unfinished.** Full-corpus shards,
token-based LR schedules, exact optimizer/RNG resume, coverage manifests,
independent tests and release criteria remain prerequisites. Saved weights
alone cannot resume production exactly. BOS-only packing needs an explicit
EOS/story-boundary review. Context 256 limits longer stories and instruction
data. See [TRAINING_PLAN.md](TRAINING_PLAN.md).

## Execution waste and the response

| Finding | Implemented response | Limit |
| --- | --- | --- |
| Dense vocabulary logits and CE copies consume memory | Optional Liger fused CE with original projection, bias and tied weight | Passed CUDA loss/gradient checks; speed depends on batch |
| Proposals used a full shifted-logit copy | Select source `t-1`; fused path projects selected sites only | Selected vocabulary probabilities still consume memory |
| Clean and RTD activation graphs overlap | Optional clean backward before RTD; one clip/update | Saving depends on backend and batch |
| Clean/generator passes computed unused RTD heads | Explicitly skip those heads | Small saving compared with backbone work |
| Decoding projected every prefix position and computed RTD | Project final position only and omit RTD | Backbone still recomputes the prefix |
| Probes and LM scoring computed unused RTD heads | Skip RTD for those operations | Dense vocabulary evaluation remains expensive |
| Benchmark read a scalar on every update | Read final detached loss after synchronization | Validation and safety checks still synchronize |
| Larger physical batches are assumed faster | Measured 64, 256 and 512; 512 added memory without speed | Updates per token change convergence |

Liger is a loss kernel here, preserving ELECTRA's LayerNorm, GELU and absolute
positions. Forced native FlashAttention preserves Q/K/V weights and configured
dropout. Installing another FlashAttention version is not improvement evidence.
AdamW fusion is explicit. Measure execution changes separately from the
self-replacement algorithm change.
The completed [optimization experiments](OPTIMIZATION.md) establish memory and
throughput improvements while leaving the quality objections above unresolved.
[Liger APIs](https://linkedin.github.io/Liger-Kernel/Low-Level-APIs/),
[PyTorch AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).

Dynamic selected-site counts and tensor-to-host branches can break `torch.compile`
graphs. Trusted packed-data paths, asynchronous loading or bounded GPU-resident
batches need boundary validation and production-loader measurements. Do not
silently remove invalid-input checks. KV caching needs positional, masking and
cached/full-prefix equivalence tests before adoption.

Compare separate joint, self joint and CLM with matched data, seeds, compute and
checkpoint selection. Monitor validation and RTD metrics, then use a frozen
independent test. Adopt self replacement for release only if cost versus quality
is competitive. Preserve failed hypotheses and the existing pilots.
