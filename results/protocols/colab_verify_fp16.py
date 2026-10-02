"""Run the four-step production FP16 resume check inside a Colab T4 kernel."""

import json
from pathlib import Path

from deletcra.verify_training import verify_training

report = verify_training("cuda", Path("/content/fp16-check"), precision="fp16")
print(json.dumps(report, indent=2), flush=True)
