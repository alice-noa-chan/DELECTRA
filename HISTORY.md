# English documentation and history cleanup

On 2026-10-01, the user explicitly requested English documentation, references,
footnotes, comments, and commit messages, and removal of Korean-language records
from Git history. README, RESEARCH, and TARGET were the only historical paths
containing Korean text. Their earlier versions were removed from all previous
commits, and their English replacements were added in a new commit.

The rewrite used git-filter-repo 2.47.0. Three Codex comparison-snapshot refs
pointed directly to trees rather than commits, so those trees were separately
rebuilt without the same document paths. Other files in the snapshots were kept.
Old reflogs and unused objects were pruned. A scan of the complete Git object
database, including unreachable objects, found no remaining Hangul text.
`git fsck --full` passed. The repository had one worktree and no configured
remote; no push or remote update was performed.

Code, measured experiment evidence, checkpoint hashes, and license files were
preserved byte for byte at the rewrite. Commit hashes changed. The
[rewrite record](results/history-rewrite-20261001.json) contains the complete
old-to-new commit map, snapshot correspondence, and preserved-file hashes.
It stores identifiers and English explanations, not the deleted document text.

Raw experiment `source_commit` fields retain the original execution-time IDs.
These IDs no longer name objects in the cleaned repository. Resolve their unique
prefix through the mapping to inspect the equivalent rewritten code commit.
The cleanup did not rerun experiments or alter their measured results.

For example, resolve the TinyStories pilot source and inspect its command code:

```python
import json
from pathlib import Path

record = json.loads(Path("results/history-rewrite-20261001.json").read_text())
original_prefix = "3a16a00"
matches = [
    new for old, new in record["commit_map"].items() if old.startswith(original_prefix)
]
assert len(matches) == 1
print(matches[0])
```

```powershell
git show <rewritten-commit>:src/deletcra/target.py
```

Development guidance now lives in [AGENTS.md](AGENTS.md). The public display name
is DELECTRA; existing `deletcra` imports, commands, and artifact paths remain
compatible. New code remains MIT-licensed, with original ELECTRA's Apache 2.0
attribution and license text retained.
