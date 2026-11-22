# Database architecture

The local SQLite database is configured with `INFERENCE_DATABASE_PATH`.
The `prediction_records` table stores outcome, model identity, latency,
creation time, actor, and prediction metadata. Successful rows include the
input checksum. Uploaded bytes and filenames are excluded. Detailed
prediction outputs and input checksums should be treated as sensitive metadata.

On first access, an idempotent transactional schema upgrade adds nullable
task, device, and error-code fields and indexes time-based history queries.
Older rows retain their original values and report unknown task metrics.

History reads support task, model, outcome, and UTC time range filters.
Researchers see their own records and aggregates. Administrators can see
records and aggregates across users. Aggregates are calculated directly from
stored inference outcomes. Average latency is calculated from completed
predictions. Failures handled by the prediction endpoint include a sanitized
HTTP error code and a request ID. Requests rejected before entering that
endpoint are outside these inference aggregates.

## PostgreSQL deployment

Install the `postgres` extra and configure `INFERENCE_DATABASE_URL` with a
`postgresql+psycopg` URL for a dedicated UTF-8 database. For a managed database,
configure certificate verification in the connection settings. Run
`python -m medical_imaging_inference.audit` once with the migration role
before starting the service. The application role needs read and insert
access to the inference table. Production requires the PostgreSQL URL.
SQLite remains available for development and preserves its existing history.
Moving existing SQLite data to PostgreSQL requires a separate data migration.

Set `INFERENCE_TEST_POSTGRES_URL` to an isolated test database and run
`pytest tests/test_postgres.py` to verify migrations and persistence.
