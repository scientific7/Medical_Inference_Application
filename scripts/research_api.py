"""Explicit loopback-only research identity adapter; never used by deployment images."""
import argparse
import os

from fastapi import HTTPException, Request
import uvicorn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("service", choices=("mlops", "inference"))
parser.add_argument("--port", type=int)
args = parser.parse_args()
if os.getenv("ENVIRONMENT") == "production" or os.getenv("INFERENCE_ENVIRONMENT") == "production":
    parser.error("Research identities are forbidden in production")


def identity(request: Request):
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        raise HTTPException(403, "Local research access only")
    return request.headers.get("X-Research-Actor", "local-researcher")


if args.service == "mlops":
    from imaging_mlops.app import app
    from imaging_mlops.auth import Principal, principal

    def researcher(request: Request):
        return Principal(identity(request), frozenset({"model_submitter", "model_reviewer", "inference_reader"}))

    app.dependency_overrides[principal] = researcher
    port = 8081
else:
    from medical_imaging_inference.app import app
    from medical_imaging_inference.security import authorize

    def researcher(request: Request):
        return {"subject": identity(request), "roles": ["inference_user", "inference_viewer", "inference_admin"]}

    app.dependency_overrides[authorize] = researcher
    port = 8082

uvicorn.run(app, host="127.0.0.1", port=args.port or port, proxy_headers=False)
