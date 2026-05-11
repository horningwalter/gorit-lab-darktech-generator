"""Colab-side bootstrap: detect GPU, mount Drive, redirect cache and output dirs.

Idempotent: safe to call multiple times. Returns a summary dict useful for the
notebook to display.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from darktech_generator.config import get_settings, reset_settings_for_tests

DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT_POINT / "MyDrive" / "gorit_lab"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def detect_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        info["available"] = torch.cuda.is_available()
        if info["available"]:
            info["name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
    except ImportError:
        info["error"] = "torch not installed"
    return info


def mount_drive() -> Path:
    if not is_colab():
        raise RuntimeError("mount_drive is only callable inside Google Colab.")
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(str(DRIVE_MOUNT_POINT), force_remount=False)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    return DRIVE_ROOT


def link_dirs_to_drive(drive_root: Path) -> dict[str, Path]:
    """Point the configured cache/output/reference dirs at Drive subfolders."""
    targets = {
        "cache": drive_root / "model_cache",
        "output": drive_root / "output_tracks",
        "reference": drive_root / "reference_tracks",
    }
    for p in targets.values():
        p.mkdir(parents=True, exist_ok=True)

    os.environ["DARKTECH_CACHE_DIR"] = str(targets["cache"])
    os.environ["DARKTECH_OUTPUT_DIR"] = str(targets["output"])
    os.environ["DARKTECH_REFERENCE_DIR"] = str(targets["reference"])
    os.environ.setdefault("HF_HOME", str(targets["cache"] / "huggingface"))
    reset_settings_for_tests()
    return targets


def free_disk_gb(path: Path = Path("/content")) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 1)


def nvidia_smi_summary() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except subprocess.SubprocessError:
        return None


def bootstrap() -> dict[str, Any]:
    """Run the full Colab setup and return a summary for the notebook to print."""
    summary: dict[str, Any] = {"is_colab": is_colab()}
    summary["gpu"] = detect_gpu()
    summary["free_disk_gb"] = free_disk_gb()

    if summary["is_colab"]:
        drive_root = mount_drive()
        summary["drive_root"] = str(drive_root)
        summary["targets"] = {k: str(v) for k, v in link_dirs_to_drive(drive_root).items()}
    else:
        settings = get_settings()
        settings.ensure_dirs()
        summary["targets"] = {
            "cache": str(settings.darktech_cache_dir),
            "output": str(settings.darktech_output_dir),
            "reference": str(settings.darktech_reference_dir),
        }

    smi = nvidia_smi_summary()
    if smi:
        summary["nvidia_smi"] = smi
    return summary
