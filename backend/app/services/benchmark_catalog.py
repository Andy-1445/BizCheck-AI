from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from app.schemas.bizscore import (
    BenchmarkCatalog,
    BenchmarkCatalogEntry,
    BenchmarkReference,
    BizScoreResult,
    PeerBenchmarkMedians,
    PeerBenchmarkRankCounts,
)
from app.schemas.company import CompanyData
from app.services.benchmark_snapshot import (
    BenchmarkSnapshotError,
    BenchmarkSnapshotStore,
)
from app.services.bizscore import (
    calculate_bizscore_total,
    classify_industry,
    score_company_age,
    score_peer_relative_position,
    score_peer_relative_position_from_rank_counts,
    score_registered_capital,
    score_registration_change_recency,
    score_registration_status,
)
from app.services.company_normalizer import calculate_company_age_years


class BenchmarkCatalogError(RuntimeError):
    """Base exception for the production Benchmark catalog."""


class BenchmarkCatalogUnavailableError(BenchmarkCatalogError):
    """The configured catalog or SQLite database cannot be read."""


class BenchmarkCatalogInvalidError(BenchmarkCatalogError):
    """Catalog data and immutable SQLite metadata do not agree."""


class BenchmarkCatalogService:
    """Resolve formal A-J snapshots and calculate a reproducible BizScore v1."""

    def __init__(
        self,
        catalog_path: str | Path,
        database_path: str | Path,
        *,
        expected_catalog_version: str,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.store = BenchmarkSnapshotStore(database_path)
        self.expected_catalog_version = expected_catalog_version
        self._catalog: BenchmarkCatalog | None = None
        self._median_cache: dict[tuple[str, str], PeerBenchmarkMedians] = {}

    @property
    def catalog(self) -> BenchmarkCatalog:
        if self._catalog is None:
            self._catalog = self._load_catalog()
        return self._catalog

    def calculate_company_bizscore(
        self,
        company: CompanyData,
        *,
        input_partial: bool = False,
        input_warnings: list[str] | None = None,
    ) -> BizScoreResult:
        catalog = self.catalog
        classification = classify_industry(company.business_items)
        age_at_snapshot = calculate_company_age_years(
            company.established_at,
            as_of=catalog.as_of,
        )

        benchmark_reference = BenchmarkReference(
            catalog_version=catalog.catalog_version,
            catalog_as_of=catalog.as_of,
            catalog_industry_mapping_version=catalog.industry_mapping_version,
            classification_version=classification.version,
        )
        peer_score = score_peer_relative_position(
            age_at_snapshot,
            company.capital.registered,
            [],
            industry_code=None,
            benchmark_version=None,
        )

        if classification.available and classification.primary_group is not None:
            industry_code = classification.primary_group.category_code
            entry = self._validated_entry(industry_code)
            benchmark_reference = BenchmarkReference(
                catalog_version=catalog.catalog_version,
                catalog_as_of=catalog.as_of,
                catalog_industry_mapping_version=catalog.industry_mapping_version,
                classification_version=classification.version,
                industry_code=industry_code,
                snapshot_version=entry.snapshot_version,
                snapshot_source_version=entry.source_version,
                snapshot_checksum_sha256=entry.checksum_sha256,
                sample_count=entry.sample_count,
            )
            rank_counts = self._rank_counts(
                entry,
                industry_code=industry_code,
                company_age_years=age_at_snapshot,
                registered_capital=company.capital.registered,
            )
            medians = self._medians(entry, industry_code=industry_code)
            peer_score = score_peer_relative_position_from_rank_counts(
                age_at_snapshot,
                company.capital.registered,
                rank_counts,
                industry_code=industry_code,
                benchmark_version=entry.snapshot_version,
                company_age_median=medians.company_age_median,
                registered_capital_median=medians.registered_capital_median,
            )

        registration_status = score_registration_status(
            company.status.code,
            company.status.description,
        )
        return calculate_bizscore_total(
            registration_status,
            score_company_age(age_at_snapshot),
            score_registered_capital(company.capital.registered),
            score_registration_change_recency(
                company.last_changed_at,
                as_of=catalog.as_of,
                established_at=company.established_at,
            ),
            peer_score,
            as_of=catalog.as_of,
            industry=classification,
            benchmark=benchmark_reference,
            input_partial=input_partial,
            input_warnings=input_warnings,
        )

    def validate_all_snapshots(self) -> BenchmarkCatalog:
        """Validate every A-J catalog entry against immutable SQLite metadata."""

        catalog = self.catalog
        for industry_code in sorted(catalog.categories):
            self._validated_entry(industry_code)
        return catalog

    def _load_catalog(self) -> BenchmarkCatalog:
        try:
            with self.catalog_path.open(encoding="utf-8") as catalog_file:
                payload = json.load(catalog_file)
            catalog = BenchmarkCatalog.model_validate(payload)
        except FileNotFoundError as exc:
            raise BenchmarkCatalogUnavailableError(
                f"Benchmark catalog was not found at {self.catalog_path}."
            ) from exc
        except OSError as exc:
            raise BenchmarkCatalogUnavailableError(
                f"Benchmark catalog could not be read at {self.catalog_path}."
            ) from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            raise BenchmarkCatalogInvalidError(
                f"Benchmark catalog is invalid: {exc}"
            ) from exc

        if catalog.catalog_version != self.expected_catalog_version:
            raise BenchmarkCatalogInvalidError(
                "Configured Benchmark catalog version does not match the catalog file: "
                f"expected {self.expected_catalog_version!r}, "
                f"found {catalog.catalog_version!r}."
            )
        return catalog

    def _validated_entry(self, industry_code: str) -> BenchmarkCatalogEntry:
        catalog = self.catalog
        entry = catalog.categories[industry_code]
        try:
            metadata = self.store.get_snapshot(entry.snapshot_version)
        except (BenchmarkSnapshotError, sqlite3.Error, OSError) as exc:
            raise BenchmarkCatalogUnavailableError(
                f"Benchmark snapshot {entry.snapshot_version!r} is unavailable."
            ) from exc

        mismatches: list[str] = []
        expected_group_sizes = {industry_code: entry.sample_count}
        if metadata.as_of != catalog.as_of:
            mismatches.append("as_of")
        if metadata.source_version != entry.source_version:
            mismatches.append("source_version")
        if metadata.industry_mapping_version != catalog.industry_mapping_version:
            mismatches.append("industry_mapping_version")
        if metadata.sample_count != entry.sample_count:
            mismatches.append("sample_count")
        if metadata.group_sample_sizes != expected_group_sizes:
            mismatches.append("group_sample_sizes")
        if metadata.checksum_sha256 != entry.checksum_sha256:
            mismatches.append("checksum_sha256")
        if mismatches:
            raise BenchmarkCatalogInvalidError(
                f"Benchmark snapshot {entry.snapshot_version!r} does not match catalog "
                f"fields: {', '.join(mismatches)}."
            )
        return entry

    def _rank_counts(
        self,
        entry: BenchmarkCatalogEntry,
        *,
        industry_code: str,
        company_age_years: float | None,
        registered_capital: int | None,
    ) -> PeerBenchmarkRankCounts:
        if company_age_years is None or registered_capital is None:
            return PeerBenchmarkRankCounts(
                sample_size=entry.sample_count,
                age_lower_count=0,
                age_equal_count=0,
                capital_lower_count=0,
                capital_equal_count=0,
            )

        try:
            counts = self.store.get_peer_rank_counts(
                entry.snapshot_version,
                industry_code,
                company_age_years=Decimal(str(company_age_years)),
                registered_capital=registered_capital,
            )
        except (BenchmarkSnapshotError, sqlite3.Error, OSError) as exc:
            raise BenchmarkCatalogUnavailableError(
                f"Benchmark samples for {entry.snapshot_version!r} are unavailable."
            ) from exc
        if counts.sample_size != entry.sample_count:
            raise BenchmarkCatalogInvalidError(
                f"Benchmark sample rows for {entry.snapshot_version!r} do not match "
                "the catalog sample_count."
            )
        return counts

    def _medians(
        self,
        entry: BenchmarkCatalogEntry,
        *,
        industry_code: str,
    ) -> PeerBenchmarkMedians:
        cache_key = (entry.snapshot_version, industry_code)
        cached = self._median_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            medians = self.store.get_peer_medians(
                entry.snapshot_version,
                industry_code,
            )
        except (BenchmarkSnapshotError, sqlite3.Error, OSError) as exc:
            raise BenchmarkCatalogUnavailableError(
                f"Benchmark medians for {entry.snapshot_version!r} are unavailable."
            ) from exc
        if medians.sample_size != entry.sample_count:
            raise BenchmarkCatalogInvalidError(
                f"Benchmark median rows for {entry.snapshot_version!r} do not match "
                "the catalog sample_count."
            )
        if (
            medians.sample_size > 0
            and (
                medians.company_age_median is None
                or medians.registered_capital_median is None
            )
        ):
            raise BenchmarkCatalogInvalidError(
                f"Benchmark medians for {entry.snapshot_version!r} are incomplete."
            )
        self._median_cache[cache_key] = medians
        return medians
