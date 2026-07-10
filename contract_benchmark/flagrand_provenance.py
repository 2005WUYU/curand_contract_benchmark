from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("flagrand_vendor.json")
HASHED_SUFFIXES = {".py", ".pt"}
REQUIRED_CURAND_SYMBOLS = (
    "create_generator",
    "generate",
    "generate_long_long",
    "generate_uniform",
    "generate_uniform_double",
    "generate_normal",
    "generate_normal_double",
    "generate_lognormal",
    "generate_lognormal_double",
    "generate_poisson",
)


def load_flagrand_vendor_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{MANIFEST_PATH} must contain a JSON object.")
    return data


def flagrand_tree_sha256(source_root: Path) -> tuple[int, str]:
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in HASHED_SUFFIXES
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return len(files), digest.hexdigest()


def flagrand_source_report(*, verify_tree: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": False,
        "manifest_path": str(MANIFEST_PATH),
    }
    try:
        manifest = load_flagrand_vendor_manifest()
        report["vendor"] = manifest
        source_root = (REPO_ROOT / str(manifest["source_path"])).resolve()
        if not source_root.is_dir():
            source_root = (REPO_ROOT / "flagrand").resolve()
        report["expected_source_root"] = str(source_root)

        import flagrand

        module_file = Path(flagrand.__file__).resolve()
        module_root = module_file.parent
        curand_module = getattr(flagrand, "curand", None)
        missing_symbols = [
            name for name in REQUIRED_CURAND_SYMBOLS if not callable(getattr(curand_module, name, None))
        ]
        report.update(
            {
                "available": True,
                "module_file": str(module_file),
                "uses_vendored_source": module_root == source_root,
                "curand_facade_available": curand_module is not None,
                "curand_facade_missing_symbols": missing_symbols,
            }
        )
        if verify_tree:
            file_count, tree_sha256 = flagrand_tree_sha256(source_root)
            report.update(
                {
                    "tree_file_count": file_count,
                    "tree_sha256": tree_sha256,
                    "tree_matches_manifest": (
                        file_count == int(manifest["tree_file_count"])
                        and tree_sha256 == str(manifest["tree_sha256"])
                    ),
                }
            )
    except BaseException as exc:
        report.update({"error": str(exc), "error_type": type(exc).__name__})
    return report


def require_vendored_flagrand() -> dict[str, Any]:
    report = flagrand_source_report(verify_tree=True)
    problems: list[str] = []
    if not report.get("available"):
        problems.append(f"FlagRand import failed: {report.get('error', 'unknown error')}")
    if not report.get("uses_vendored_source"):
        problems.append(
            "FlagRand resolved outside the vendored source root: "
            f"module={report.get('module_file')} expected={report.get('expected_source_root')}"
        )
    if not report.get("curand_facade_available"):
        problems.append("flagrand.curand is unavailable")
    missing = report.get("curand_facade_missing_symbols") or []
    if missing:
        problems.append(f"flagrand.curand is missing required symbols: {missing}")
    if not report.get("tree_matches_manifest"):
        problems.append(
            "Vendored FlagRand tree does not match flagrand_vendor.json: "
            f"files={report.get('tree_file_count')} sha256={report.get('tree_sha256')}"
        )
    if problems:
        raise RuntimeError("; ".join(problems))
    return report
