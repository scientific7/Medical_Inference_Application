# Synthetic four-service acceptance test

Run with a Python 3.10 environment containing the services' test dependencies,
PyTorch and TensorFlow, from the contracts checkout:

```bash
python scripts/test_platform.py --workspace /path/to/sibling/checkouts
```

Explicit `--dataset`, `--models`, `--mlops`, and `--inference` paths override
folder naming. The test calls the actual Dataset Pipeline publisher and Model
Engineering reader/trainer/exporter, then actual MLOps and inference ASGI routes.
Only authentication identities and HTTP transport are supplied by the test.
No AWS requests, real datasets, production approvals, or public servers are used.
All files and SQLite databases are isolated in pytest temporary directories.

Classification exercises both PyTorch and TensorFlow. Segmentation and detection
exercise publisher/consumer handoff including verified-negative images. Failure
cases cover tampered manifests, duplicate image leakage, missing assets,
self-approval, changed packages, and unapproved inference. This is a local
integration gate; it does not prove hosted SSO, AWS deployment or throughput.
