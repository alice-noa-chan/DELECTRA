# Project guidance

## Language and explanation

- Write documentation, footnotes, citations, comments, docstrings, and commit
  messages in English. Conversation language can follow the user.
- Use DELECTRA as the display name. Existing `deletcra` imports, CLI names, and
  saved artifact paths remain valid; avoid breaking them during unrelated work.
- Keep code responsibilities clear: configuration, models, objectives, data,
  training/evaluation, and command-line orchestration belong in separate modules.
- Explain important assumptions and non-obvious steps in simple English. Include
  tensor shapes, token alignment, and gradient paths when they prevent mistakes.
  Do not bury straightforward code in repetitive comments or needless helpers.

## Model identity and research evidence

- Preserve the ELECTRA backbone, discriminator head, and replaced-token detection
  as the main research direction. CLM-only is a control for measuring RTD's value.
- Document departures from original ELECTRA, including causal attention,
  next-token generator training, additional CLM losses, and hyperparameters.
- Keep measured results, source provenance, and limitations intact. Distinguish
  unique data from repeated input tokens, final from selected checkpoints, and
  validation from independent test evidence.
- Keep new code under MIT and preserve original ELECTRA's Apache 2.0 attribution,
  NOTICE, and upstream license text. External data/models retain their licenses.

## Change workflow

1. Edit the code and relevant documentation.
2. Run Ruff formatting and linting.
3. Run tests appropriate to the change, then the required project checks.
4. Make separate coherent commits with detailed English messages describing the
   reason, behavior, and validation.

Use branch names without a `codex/` prefix, as requested by the user. Rewrite Git
history only when the user explicitly requests it. When rewriting recorded
experiment commits, preserve an old-to-new hash mapping for reproducibility.
