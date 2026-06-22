from __future__ import annotations

import os
from pathlib import Path


def configure_writable_cache(workspace_root: Path, *, shard: str | int | None = None) -> dict[str, str]:
    """Keep CUDA, Triton, and extension caches out of unwritable root paths."""
    root = workspace_root.resolve()
    shard_text = _safe_name(str(shard)) if shard is not None else _safe_name(os.environ.get("CURAND_CONTRACT_SHARD", "default"))

    home = os.environ.get("HOME", "")
    if not home or home == "/" or not _ensure_writable(Path(home)):
        os.environ["HOME"] = str(root)

    default_xdg = root / ".cache"
    xdg_env = os.environ.get("XDG_CACHE_HOME")
    xdg = Path(xdg_env or default_xdg)
    if _looks_like_root_cache(xdg) or not _ensure_writable(xdg):
        xdg = default_xdg
        _ensure_writable(xdg)
    if not xdg_env or Path(os.environ.get("XDG_CACHE_HOME", "")) != xdg:
        os.environ["XDG_CACHE_HOME"] = str(xdg)

    default_triton = xdg / "triton" / shard_text
    triton_env = os.environ.get("TRITON_CACHE_DIR")
    triton_cache = Path(triton_env or default_triton)
    if _looks_like_root_cache(triton_cache) or not _ensure_writable(triton_cache):
        triton_cache = default_triton
        _ensure_writable(triton_cache)
    if not triton_env or Path(os.environ.get("TRITON_CACHE_DIR", "")) != triton_cache:
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)

    torch_extensions = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not torch_extensions:
        torch_extensions_path = xdg / "torch_extensions"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_path)
        _ensure_writable(torch_extensions_path)

    return {
        "HOME": os.environ.get("HOME", ""),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", ""),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
    }


def _ensure_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _looks_like_root_cache(path: Path) -> bool:
    text = str(path)
    return text in {"/.triton", "/.cache", "/torch_extensions"} or text.startswith("/.triton/")


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in value)
    return text or "default"
