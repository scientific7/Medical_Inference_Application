# Component architecture

Streamlit reads registry, runtime, evaluation, and audit data through
FastAPI. Inside the backend, the registry adapter handles discovery
and approval. The storage adapter resolves approved content. The runtime
cache holds loaded framework exports. Preprocessing and postprocessing
adapt the package and task contracts. The audit module stores prediction
metadata in PostgreSQL or SQLite.

[Component diagram](../diagrams/component-diagram.mmd)
