from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from contract_benchmark.flagrand_provenance import require_vendored_flagrand  # noqa: E402


def main() -> int:
    report = require_vendored_flagrand()
    vendor = report["vendor"]
    print(
        "[flagrand-preflight] "
        f"commit={vendor['commit']} api={vendor['api_surface']} module={report['module_file']}"
    )
    print(
        "[flagrand-preflight] "
        f"files={report['tree_file_count']} sha256={report['tree_sha256']} verified=true"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
