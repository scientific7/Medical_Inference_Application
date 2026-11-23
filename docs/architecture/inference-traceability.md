# Local inference requirement traceability

| Requirements | Existing implementation | Evidence and gaps |
| --- | --- | --- |
| 1–4, 19 P0 | FastAPI, registry adapters, private package verifier, runtime cache | Current Layer 3 publishes an approved catalogue and a projected exact-version response. Older inference adapters expected the internal record. Verify both forms in registry tests. |
| 5–7 | Multipart upload, model-driven preprocessing, TorchScript and SavedModel adapters, bounded batch queue | Synthetic validation and tensor tests exist. Real framework and device coverage depends on installed runtimes. |
| 8–11 | Task decoders and Streamlit results | Decoder and overlay tests exist. Raw detection heads need a supported export. |
| 12–14 | SQLite/PostgreSQL audit, history, structured events, health, readiness and metrics | Persistence and access tests exist. Extended latency and resource telemetry remain incomplete. |
| 15–17 | Streamlit workstation and loopback launcher | Local launcher supplies a research identity override. Production uses OIDC. |
| 18, 20 | API tests, Streamlit tests and shared-contract integration suite | Run the local lifecycle with synthetic exports and identify unavailable framework or upstream checks separately. |

## Classification lifecycle evidence

`tests/test_local_lifecycle.py` generates real TorchScript and TensorFlow
SavedModel exports. It exercises the current governance HTTP response
contract, approved discovery, verified model cards, PNG upload,
preprocessing, model execution, scores, SQLite history, and deprecation
checks. The upstream HTTP boundary is mocked. The Streamlit tests exercise
UI components separately. Live Layer 3 registration remains a separate
integration check.

Successful history records include input format and dimensions, preprocessing
version, and application version. Approval time is retained when governance
includes it. Model-version filtering is available in the API and workstation.
See the [current milestone evidence](../../IMPLEMENTATION_STATUS.md).
