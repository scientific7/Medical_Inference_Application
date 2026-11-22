# Contributing

Use Python 3.10 and install `.[test,frontend,aws,postgres]`. Run
`pytest -q`, `ruff check src apps tests`, and
`ruff format --check src apps tests`. Exercise database changes against
an isolated PostgreSQL instance. Use synthetic images and model outputs
in tests. Keep medical images, checkpoints, and credentials outside Git.
