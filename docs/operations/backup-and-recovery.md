# Backup and recovery

Use PostgreSQL backups or managed snapshots and test restores in an isolated
database. Encrypt backups and limit read access to approved operators.
SQLite backups can use SQLite's backup API to capture a consistent database.
Record the deployed image digests and approved artifact prefixes separately.

After restoration, verify schema version compatibility, read a known
requester's history, and run an approved-model smoke test. Batch status and
runtime cache entries are process-local and are recreated after restart.
