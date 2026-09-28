# Benchmark snapshot storage

Versioned SQLite Benchmark snapshots are created here at build or deployment
time. Database files are intentionally ignored by Git because they are derived
from dated GCIS exports and may be large.

Do not place fabricated company rows in a production snapshot. Each snapshot
must retain its source version, reference date, industry-mapping version, sample
counts, exclusion counts, and checksum.

The local `bizcheck-benchmark.sqlite3` contains immutable nationwide A-J
category snapshots for the 2026-08-01 reference date. Their production mapping,
sample counts, source versions, and checksums are recorded in
`benchmark-catalog-2026-08-01-v1.json`; per-snapshot metadata files remain the
audit records. `benchmark-catalog-2026-08-01-v1-verification.json` records the
full release fingerprint and count verification. The SQLite database is ignored
by Git.
