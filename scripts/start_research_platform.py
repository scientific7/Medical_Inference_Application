"""Start dataset, governance and inference APIs/UIs on loopback for research."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workspace", type=Path, default=root.parent)
parser.add_argument("--check", action="store_true", help="Validate environments and ports without starting services")
args = parser.parse_args()
dataset = args.workspace / "medical_dataset_pipeline"
mlops = args.workspace / "MLOps-and-model-governance"


def python_for(repo, modules):
    for name in (".venv310", ".venv-impl", ".venv"):
        candidate = repo / name / "bin/python"
        if candidate.exists() and subprocess.run([str(candidate), "-c", modules],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return str(candidate)
    parser.error(f"Install the documented extras in a virtual environment under {repo}")


python_dataset = python_for(dataset, "import alembic, streamlit, fastapi")
python_mlops = python_for(mlops, "import uvicorn, streamlit, medical_imaging_contracts")
python_inference = python_for(root, "import uvicorn, streamlit, medical_imaging_contracts")
for port in (8000, 8081, 8082, 8501, 8502, 8503):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            parser.error(f"Port {port} is already in use; stop the existing local service first")
if args.check:
    print("Research environment and six loopback ports are available")
    raise SystemExit(0)

env = dict(os.environ)
env.update(ENVIRONMENT="development", INFERENCE_ENVIRONMENT="development",
           MLOPS_API_URL="http://127.0.0.1:8081", MLOPS_SERVICE_TOKEN="local-research-service",
           INFERENCE_API_URL="http://127.0.0.1:8082", INFERENCE_DATASET_API_URL="http://127.0.0.1:8000")
# A shared private local artifact mount; no dataset copying is performed.
artifacts = str(mlops / "artifacts")
env.setdefault("MLOPS_ARTIFACT_ROOT", artifacts)
env.setdefault("INFERENCE_ARTIFACT_ROOT", artifacts)
env["PYTHONPATH"] = os.pathsep.join(str(path) for path in (
    dataset, dataset / "src", mlops / "src", root / "src", root / "packages/contracts/src"))
commands = [
    (["bash", str(dataset / "scripts/start_platform_dashboard.sh")], dataset, dict(env, PYTHON=python_dataset)),
    ([python_mlops, str(root / "scripts/research_api.py"), "mlops"], mlops, env),
    ([python_inference, str(root / "scripts/research_api.py"), "inference"], root, env),
    ([python_mlops, "-m", "streamlit", "run", "dashboard/app.py", "--server.address=127.0.0.1", "--server.port=8502", "--server.headless=true"], mlops, env),
    ([python_inference, "-m", "streamlit", "run", "apps/frontend/app.py", "--server.address=127.0.0.1", "--server.port=8503", "--server.headless=true"], root, env),
]
children = []
try:
    for command, cwd, child_env in commands:
        children.append(subprocess.Popen(command, cwd=cwd, env=child_env, start_new_session=True))
    print("Datasets: http://127.0.0.1:8501 | Governance: http://127.0.0.1:8502 | Inference: http://127.0.0.1:8503", flush=True)
    while all(child.poll() is None for child in children):
        time.sleep(0.5)
    raise SystemExit("A service exited; review its output above")
except KeyboardInterrupt:
    pass
finally:
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
