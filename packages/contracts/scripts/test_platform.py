"""Run opt-in synthetic integration against explicitly selected sibling checkouts."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, default=root.parents[2])
parser.add_argument("--dataset", type=Path)
parser.add_argument("--models", type=Path)
parser.add_argument("--mlops", type=Path)
parser.add_argument("--inference", type=Path)
args = parser.parse_args()
repos = {
    "dataset": args.dataset or args.workspace / "medical_dataset_pipeline",
    "models": args.models or args.workspace / "medical_imaging_model",
    "mlops": args.mlops or args.workspace / "MLOps-and-model-governance",
    "inference": args.inference or args.workspace / "medical-imaging-inference-platform",
}
for name, path in repos.items():
    if not (path / "src").is_dir():
        parser.error(f"Missing {name} checkout: {path}; supply --{name}")
env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p / "src") for p in [root, *repos.values()]))
env["PLATFORM_SYNTHETIC_TEST"] = "1"
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "--import-mode=importlib",
                                  str(root / "integration")], env=env, cwd=root))
