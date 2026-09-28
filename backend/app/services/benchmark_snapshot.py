from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from app.schemas.bizscore import (
    PeerBenchmarkMedians,
    BenchmarkSnapshotMetadata,
    PeerBenchmarkRankCounts,
    PeerBenchmarkSample,
)
from app.schemas.company import CompanyData, CompanyResponse
from app.services.bizscore import INDUSTRY_MAPPING_VERSION, classify_industry
from app.services.company_normalizer import calculate_company_age_years


SNAPSHOT_SCHEMA_VERSION = 1
BENCHMARK_ELIGIBLE_INDUSTRY_CODES = frozenset("ABCDEFGHIJ")


class BenchmarkSnapshotError(RuntimeError):
    """Base exception for versioned Benchmark snapshot operations."""


class BenchmarkSnapshotExistsError(BenchmarkSnapshotError):
    """The requested immutable snapshot version already exists."""


class BenchmarkSnapshotNotFoundError(BenchmarkSnapshotError):
    """The requested snapshot version is not available."""


class BenchmarkSnapshotInputError(BenchmarkSnapshotError):
    """The snapshot source is invalid or would create ambiguous results."""


class BenchmarkSnapshotStore:
    """Create and read immutable, versioned peer Benchmark snapshots."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_snapshots (
                    version TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    industry_mapping_version TEXT NOT NULL,
                    input_record_count INTEGER NOT NULL CHECK (input_record_count >= 0),
                    sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
                    excluded_record_count INTEGER NOT NULL
                        CHECK (excluded_record_count >= 0),
                    exclusion_counts_json TEXT NOT NULL,
                    group_sample_sizes_json TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL
                        CHECK (length(checksum_sha256) = 64)
                );

                CREATE TABLE IF NOT EXISTS benchmark_samples (
                    snapshot_version TEXT NOT NULL,
                    tax_id TEXT NOT NULL
                        CHECK (length(tax_id) = 8 AND tax_id NOT GLOB '*[^0-9]*'),
                    industry_code TEXT NOT NULL
                        CHECK (industry_code IN ('A','B','C','D','E','F','G','H','I','J')),
                    established_at TEXT NOT NULL,
                    company_age_years TEXT NOT NULL,
                    registered_capital INTEGER NOT NULL
                        CHECK (registered_capital >= 0),
                    PRIMARY KEY (snapshot_version, tax_id),
                    FOREIGN KEY (snapshot_version)
                        REFERENCES benchmark_snapshots(version)
                        ON UPDATE RESTRICT ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_benchmark_samples_group
                    ON benchmark_samples(snapshot_version, industry_code);
                """
            )
            connection.execute(f"PRAGMA user_version = {SNAPSHOT_SCHEMA_VERSION}")

    def create_snapshot(
        self,
        companies: Iterable[CompanyData],
        *,
        version: str,
        as_of: date,
        source: str,
        source_version: str,
        created_at: datetime | None = None,
        industry_code_override: str | None = None,
        industry_mapping_version: str = INDUSTRY_MAPPING_VERSION,
    ) -> BenchmarkSnapshotMetadata:
        normalized_version = _required_text(version, "version")
        normalized_source = _required_text(source, "source")
        normalized_source_version = _required_text(
            source_version,
            "source_version",
        )
        normalized_mapping_version = _required_text(
            industry_mapping_version,
            "industry_mapping_version",
        )
        normalized_industry_override = (
            _normalize_industry_code(industry_code_override)
            if industry_code_override is not None
            else None
        )
        if not isinstance(as_of, date) or isinstance(as_of, datetime):
            raise BenchmarkSnapshotInputError("as_of must be a date.")

        snapshot_created_at = created_at or datetime.now(timezone.utc)
        if (
            not isinstance(snapshot_created_at, datetime)
            or snapshot_created_at.tzinfo is None
            or snapshot_created_at.utcoffset() is None
        ):
            raise BenchmarkSnapshotInputError(
                "created_at must be a timezone-aware datetime."
            )

        rows, input_count, exclusion_counts = _prepare_snapshot_rows(
            companies,
            as_of=as_of,
            industry_code_override=normalized_industry_override,
        )
        if not rows:
            raise BenchmarkSnapshotInputError(
                "Snapshot contains no eligible Benchmark records."
            )
        group_sample_sizes = dict(
            sorted(Counter(row.industry_code for row in rows).items())
        )
        checksum = _snapshot_checksum(rows)
        excluded_count = sum(exclusion_counts.values())
        metadata = BenchmarkSnapshotMetadata(
            version=normalized_version,
            as_of=as_of,
            created_at=snapshot_created_at,
            source=normalized_source,
            source_version=normalized_source_version,
            industry_mapping_version=normalized_mapping_version,
            input_record_count=input_count,
            sample_count=len(rows),
            excluded_record_count=excluded_count,
            exclusion_counts=dict(sorted(exclusion_counts.items())),
            group_sample_sizes=group_sample_sizes,
            checksum_sha256=checksum,
        )

        self.initialize()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO benchmark_snapshots (
                        version,
                        as_of,
                        created_at,
                        source,
                        source_version,
                        industry_mapping_version,
                        input_record_count,
                        sample_count,
                        excluded_record_count,
                        exclusion_counts_json,
                        group_sample_sizes_json,
                        checksum_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.version,
                        metadata.as_of.isoformat(),
                        metadata.created_at.isoformat(),
                        metadata.source,
                        metadata.source_version,
                        metadata.industry_mapping_version,
                        metadata.input_record_count,
                        metadata.sample_count,
                        metadata.excluded_record_count,
                        json.dumps(
                            metadata.exclusion_counts,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(
                            metadata.group_sample_sizes,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        metadata.checksum_sha256,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO benchmark_samples (
                        snapshot_version,
                        tax_id,
                        industry_code,
                        established_at,
                        company_age_years,
                        registered_capital
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            metadata.version,
                            row.tax_id,
                            row.industry_code,
                            row.established_at.isoformat(),
                            _format_decimal(row.company_age_years),
                            row.registered_capital,
                        )
                        for row in rows
                    ],
                )
        except sqlite3.IntegrityError as exc:
            if self.get_snapshot(normalized_version, required=False) is not None:
                raise BenchmarkSnapshotExistsError(
                    f"Benchmark snapshot version {normalized_version!r} already exists."
                ) from exc
            raise BenchmarkSnapshotError("Benchmark snapshot could not be stored.") from exc

        return metadata

    def get_snapshot(
        self,
        version: str,
        *,
        required: bool = True,
    ) -> BenchmarkSnapshotMetadata | None:
        normalized_version = _required_text(version, "version")
        if not self.database_path.exists():
            if required:
                raise BenchmarkSnapshotNotFoundError(
                    f"Benchmark snapshot version {normalized_version!r} was not found."
                )
            return None

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM benchmark_snapshots WHERE version = ?",
                (normalized_version,),
            ).fetchone()
        if row is None:
            if required:
                raise BenchmarkSnapshotNotFoundError(
                    f"Benchmark snapshot version {normalized_version!r} was not found."
                )
            return None
        return _metadata_from_row(row)

    def get_latest_snapshot(self) -> BenchmarkSnapshotMetadata:
        if not self.database_path.exists():
            raise BenchmarkSnapshotNotFoundError("No Benchmark snapshots are available.")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM benchmark_snapshots
                ORDER BY as_of DESC, created_at DESC, version DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise BenchmarkSnapshotNotFoundError("No Benchmark snapshots are available.")
        return _metadata_from_row(row)

    def get_peer_samples(
        self,
        version: str,
        industry_code: str,
    ) -> list[PeerBenchmarkSample]:
        normalized_version = _required_text(version, "version")
        normalized_industry_code = _normalize_industry_code(industry_code)
        self.get_snapshot(normalized_version)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT company_age_years, registered_capital
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                ORDER BY tax_id
                """,
                (normalized_version, normalized_industry_code),
            ).fetchall()
        return [
            PeerBenchmarkSample(
                company_age_years=Decimal(row["company_age_years"]),
                registered_capital=row["registered_capital"],
            )
            for row in rows
        ]

    def count_peer_samples(self, version: str, industry_code: str) -> int:
        normalized_version = _required_text(version, "version")
        normalized_industry_code = _normalize_industry_code(industry_code)
        self.get_snapshot(normalized_version)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS sample_count
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                """,
                (normalized_version, normalized_industry_code),
            ).fetchone()
        return int(row["sample_count"])

    def get_peer_rank_counts(
        self,
        version: str,
        industry_code: str,
        *,
        company_age_years: Decimal,
        registered_capital: int,
    ) -> PeerBenchmarkRankCounts:
        """Return midrank inputs without loading an entire peer group into memory."""

        normalized_version = _required_text(version, "version")
        normalized_industry_code = _normalize_industry_code(industry_code)
        if (
            not isinstance(company_age_years, Decimal)
            or not company_age_years.is_finite()
            or company_age_years < 0
        ):
            raise BenchmarkSnapshotInputError(
                "company_age_years must be a finite non-negative Decimal."
            )
        if (
            isinstance(registered_capital, bool)
            or not isinstance(registered_capital, int)
            or registered_capital < 0
        ):
            raise BenchmarkSnapshotInputError(
                "registered_capital must be a non-negative integer."
            )

        self.get_snapshot(normalized_version)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS sample_size,
                    COALESCE(SUM(
                        CAST(company_age_years AS REAL) < CAST(? AS REAL)
                    ), 0) AS age_lower_count,
                    COALESCE(SUM(
                        CAST(company_age_years AS REAL) = CAST(? AS REAL)
                    ), 0) AS age_equal_count,
                    COALESCE(SUM(registered_capital < ?), 0)
                        AS capital_lower_count,
                    COALESCE(SUM(registered_capital = ?), 0)
                        AS capital_equal_count
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                """,
                (
                    str(company_age_years),
                    str(company_age_years),
                    registered_capital,
                    registered_capital,
                    normalized_version,
                    normalized_industry_code,
                ),
            ).fetchone()

        return PeerBenchmarkRankCounts(
            sample_size=int(row["sample_size"]),
            age_lower_count=int(row["age_lower_count"]),
            age_equal_count=int(row["age_equal_count"]),
            capital_lower_count=int(row["capital_lower_count"]),
            capital_equal_count=int(row["capital_equal_count"]),
        )

    def get_peer_medians(
        self,
        version: str,
        industry_code: str,
    ) -> PeerBenchmarkMedians:
        """Return age and registered-capital medians for one snapshot group."""

        normalized_version = _required_text(version, "version")
        normalized_industry_code = _normalize_industry_code(industry_code)
        self.get_snapshot(normalized_version)
        with self._connect() as connection:
            count_row = connection.execute(
                """
                SELECT COUNT(*) AS sample_size
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                """,
                (normalized_version, normalized_industry_code),
            ).fetchone()
            sample_size = int(count_row["sample_size"])
            if sample_size == 0:
                return PeerBenchmarkMedians(sample_size=0)

            middle_offset = (sample_size - 1) // 2
            middle_row_count = 1 if sample_size % 2 else 2
            age_rows = connection.execute(
                """
                SELECT company_age_years AS value
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                ORDER BY CAST(company_age_years AS REAL), tax_id
                LIMIT ? OFFSET ?
                """,
                (
                    normalized_version,
                    normalized_industry_code,
                    middle_row_count,
                    middle_offset,
                ),
            ).fetchall()
            capital_rows = connection.execute(
                """
                SELECT registered_capital AS value
                FROM benchmark_samples
                WHERE snapshot_version = ? AND industry_code = ?
                ORDER BY registered_capital, tax_id
                LIMIT ? OFFSET ?
                """,
                (
                    normalized_version,
                    normalized_industry_code,
                    middle_row_count,
                    middle_offset,
                ),
            ).fetchall()

        return PeerBenchmarkMedians(
            sample_size=sample_size,
            company_age_median=_median_decimal(
                [Decimal(row["value"]) for row in age_rows]
            ),
            registered_capital_median=_median_decimal(
                [Decimal(row["value"]) for row in capital_rows]
            ),
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class _SnapshotRow:
    __slots__ = (
        "tax_id",
        "industry_code",
        "established_at",
        "company_age_years",
        "registered_capital",
    )

    def __init__(
        self,
        *,
        tax_id: str,
        industry_code: str,
        established_at: date,
        company_age_years: Decimal,
        registered_capital: int,
    ) -> None:
        self.tax_id = tax_id
        self.industry_code = industry_code
        self.established_at = established_at
        self.company_age_years = company_age_years
        self.registered_capital = registered_capital


def load_companies_from_jsonl(path: str | Path) -> list[CompanyData]:
    """Load canonical CompanyData or CompanyResponse objects from JSON Lines."""

    return list(iter_companies_from_jsonl(path))


def iter_companies_from_jsonl(path: str | Path) -> Iterator[CompanyData]:
    """Stream canonical CompanyData or CompanyResponse objects from JSON Lines."""

    input_path = Path(path)
    with input_path.open(encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSON value must be an object")
                if "data" in payload and "meta" in payload:
                    company = CompanyResponse.model_validate(payload).data
                else:
                    company = CompanyData.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                raise BenchmarkSnapshotInputError(
                    f"Invalid JSONL record at line {line_number}: {exc}"
                ) from exc
            yield company


def _prepare_snapshot_rows(
    companies: Iterable[CompanyData],
    *,
    as_of: date,
    industry_code_override: str | None,
) -> tuple[list[_SnapshotRow], int, Counter[str]]:
    rows: list[_SnapshotRow] = []
    input_count = 0
    exclusion_counts: Counter[str] = Counter()
    seen_tax_ids: set[str] = set()

    for company in companies:
        input_count += 1
        if not isinstance(company, CompanyData):
            raise BenchmarkSnapshotInputError(
                f"Input record {input_count} must be CompanyData."
            )

        tax_id = company.tax_id.strip()
        if len(tax_id) != 8 or not tax_id.isascii() or not tax_id.isdigit():
            exclusion_counts["invalid_tax_id"] += 1
            continue
        if tax_id in seen_tax_ids:
            raise BenchmarkSnapshotInputError(
                f"Duplicate tax_id {tax_id!r} would make the snapshot ambiguous."
            )
        seen_tax_ids.add(tax_id)

        status_code = company.status.code
        if not isinstance(status_code, str) or status_code.strip() != "01":
            exclusion_counts["status_not_active"] += 1
            continue
        established_at = company.established_at
        age = calculate_company_age_years(established_at, as_of=as_of)
        if established_at is None or age is None:
            exclusion_counts["invalid_established_at"] += 1
            continue

        registered_capital = company.capital.registered
        if (
            registered_capital is None
            or isinstance(registered_capital, bool)
            or registered_capital < 0
        ):
            exclusion_counts["invalid_registered_capital"] += 1
            continue

        if industry_code_override is not None:
            industry_code = industry_code_override
        else:
            classification = classify_industry(company.business_items)
            if not classification.available or classification.primary_group is None:
                exclusion_counts["industry_unavailable"] += 1
                continue
            industry_code = classification.primary_group.category_code
            if industry_code not in BENCHMARK_ELIGIBLE_INDUSTRY_CODES:
                exclusion_counts["industry_unavailable"] += 1
                continue

        rows.append(
            _SnapshotRow(
                tax_id=tax_id,
                industry_code=industry_code,
                established_at=established_at,
                company_age_years=Decimal(str(age)),
                registered_capital=registered_capital,
            )
        )

    rows.sort(key=lambda row: row.tax_id)
    return rows, input_count, exclusion_counts


def _snapshot_checksum(rows: list[_SnapshotRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row.tax_id}|{row.industry_code}|{row.established_at.isoformat()}|"
                f"{_format_decimal(row.company_age_years)}|{row.registered_capital}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _metadata_from_row(row: sqlite3.Row) -> BenchmarkSnapshotMetadata:
    return BenchmarkSnapshotMetadata(
        version=row["version"],
        as_of=date.fromisoformat(row["as_of"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        source=row["source"],
        source_version=row["source_version"],
        industry_mapping_version=row["industry_mapping_version"],
        input_record_count=row["input_record_count"],
        sample_count=row["sample_count"],
        excluded_record_count=row["excluded_record_count"],
        exclusion_counts=json.loads(row["exclusion_counts_json"]),
        group_sample_sizes=json.loads(row["group_sample_sizes_json"]),
        checksum_sha256=row["checksum_sha256"],
    )


def _normalize_industry_code(value: object) -> str:
    if not isinstance(value, str):
        raise BenchmarkSnapshotInputError("industry_code must be A-J.")
    code = value.strip().upper()
    if code not in BENCHMARK_ELIGIBLE_INDUSTRY_CODES:
        raise BenchmarkSnapshotInputError("industry_code must be A-J.")
    return code


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkSnapshotInputError(f"{field} must be a non-empty string.")
    return value.strip()


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def _median_decimal(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, start=Decimal(0)) / Decimal(len(values))
