from __future__ import annotations

import base64
import gzip
import hashlib
import io
import joblib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "build_assets"
WORK = ROOT / ".v5_build"
EXPECTED_PRELAUNCH_SHA256 = "e4fc92bbff264aab4cc3aa67f91c88edddea5d308bbaad33c6e24adb837d15cf"
FREEZE_MANIFEST_SHA256 = "5f3874c8a663d329d7e19876e924f44ee09dd0683ac72dc5a017811e05125792"
OBJECTIVES = ["Sales", "Traffic", "Awareness", "Lead Generation", "Engagement", "App Installs"]


def joined(prefix: str) -> bytes:
    files = sorted(PARTS.glob(f"{prefix}_*.txt"))
    if not files:
        raise RuntimeError(f"Missing build asset chunks: {prefix}")
    return base64.b64decode("".join(p.read_text().strip() for p in files))


def main() -> None:
    # Restore templates/static/metadata that are kept as one compressed text artifact.
    with zipfile.ZipFile(io.BytesIO(joined("assets"))) as zf:
        zf.extractall(ROOT)

    WORK.mkdir(exist_ok=True)
    generator = gzip.decompress(joined("gen")).decode("utf-8")
    trainer = gzip.decompress(joined("train")).decode("utf-8")
    generator = generator.replace("out=Path('/mnt/data')", f"out=Path({str(WORK)!r})")
    trainer = trainer.replace("BASE=Path('/mnt/data')", f"BASE=Path({str(WORK)!r})")
    (WORK / "generate.py").write_text(generator)
    (WORK / "train.py").write_text(trainer)

    subprocess.run([sys.executable, str(WORK / "generate.py")], check=True)
    prelaunch = WORK / "saudi_marketing_campaigns_v5_prelaunch.csv"
    got = hashlib.sha256(prelaunch.read_bytes()).hexdigest()
    if got != EXPECTED_PRELAUNCH_SHA256:
        raise RuntimeError(f"V5 deterministic dataset hash mismatch: {got}")

    models = {}
    for objective in OBJECTIVES:
        subprocess.run([sys.executable, str(WORK / "train.py"), objective], check=True)
        slug = objective.replace(" ", "_").lower()
        models[objective] = joblib.load(WORK / f"v5_model_{slug}.joblib")

    suite = {
        "dataset_version": "V5",
        "freeze_manifest_sha256": FREEZE_MANIFEST_SHA256,
        "models": models,
        "lockbox_results_file": "saudi_marketing_v5_final_lockbox_results.json",
    }
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(suite, model_dir / "model_suite.joblib", compress=("lzma", 9))

    # Keep deployment lean; V5 is reconstructed deterministically on every build.
    shutil.rmtree(WORK, ignore_errors=True)
    print("MarketPredict V5 deterministic model build complete.")


if __name__ == "__main__":
    main()
