from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit


BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def parse_cors_origins(raw_value: str) -> tuple[str, ...]:
    """Parse an exact-origin CORS allowlist and reject unsafe URL shapes."""

    origins: list[str] = []
    for raw_origin in raw_value.split(","):
        origin = raw_origin.strip()
        if not origin:
            continue
        if origin == "*":
            raise ValueError("CORS wildcard origins are not allowed.")
        if any(character.isspace() for character in origin) or "\\" in origin:
            raise ValueError(f"Invalid characters in CORS origin: {origin}")

        parsed = urlsplit(origin)
        try:
            parsed.port
        except ValueError as error:
            raise ValueError(f"Invalid CORS origin port: {origin}") from error
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"CORS origin must use http or https: {origin}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"CORS origin must not contain credentials: {origin}")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError(
                f"CORS origin must not contain a path, query, or fragment: {origin}"
            )
        if "*" in parsed.hostname:
            raise ValueError(f"CORS origin must be an exact host: {origin}")

        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        default_port = (
            parsed.scheme.lower() == "http" and port == 80
        ) or (
            parsed.scheme.lower() == "https" and port == 443
        )
        normalized = f"{parsed.scheme.lower()}://{host}"
        if port is not None and not default_port:
            normalized += f":{port}"
        if normalized not in origins:
            origins.append(normalized)
    return tuple(origins)


def _cors_origins_env() -> tuple[str, ...]:
    raw_value = os.getenv("BIZCHECK_CORS_ORIGINS")
    if raw_value is None:
        return DEFAULT_CORS_ORIGINS
    return parse_cors_origins(raw_value)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "BizCheck AI API"
    app_version: str = "0.1.0"
    environment: str = os.getenv("BIZCHECK_ENVIRONMENT", "development")
    cors_origins: tuple[str, ...] = field(default_factory=_cors_origins_env)
    gcis_base_url: str = os.getenv(
        "BIZCHECK_GCIS_BASE_URL",
        "https://data.gcis.nat.gov.tw/od/data/api",
    )
    gcis_timeout_seconds: float = float(
        os.getenv("BIZCHECK_GCIS_TIMEOUT_SECONDS", "10")
    )
    gcis_max_retries: int = int(os.getenv("BIZCHECK_GCIS_MAX_RETRIES", "1"))
    gcis_retry_backoff_seconds: float = float(
        os.getenv("BIZCHECK_GCIS_RETRY_BACKOFF_SECONDS", "0.2")
    )
    gcis_probe_timeout_seconds: float = float(
        os.getenv("BIZCHECK_GCIS_PROBE_TIMEOUT_SECONDS", "2")
    )
    gcis_cache_max_entries: int = int(
        os.getenv("BIZCHECK_GCIS_CACHE_MAX_ENTRIES", "256")
    )
    gcis_cache_ttl_seconds: int = int(
        os.getenv("BIZCHECK_GCIS_CACHE_TTL_SECONDS", "300")
    )
    gcis_cache_stale_if_error_seconds: int = int(
        os.getenv("BIZCHECK_GCIS_CACHE_STALE_IF_ERROR_SECONDS", "21600")
    )
    benchmark_catalog_version: str = os.getenv(
        "BIZCHECK_BENCHMARK_CATALOG_VERSION",
        "benchmark-catalog-2026-08-01-v1",
    )
    benchmark_catalog_path: Path = Path(
        os.getenv(
            "BIZCHECK_BENCHMARK_CATALOG_PATH",
            str(
                BACKEND_ROOT
                / "data"
                / "benchmarks"
                / "benchmark-catalog-2026-08-01-v1.json"
            ),
        )
    )
    benchmark_database_path: Path = Path(
        os.getenv(
            "BIZCHECK_BENCHMARK_DATABASE_PATH",
            str(BACKEND_ROOT / "data" / "benchmarks" / "bizcheck-benchmark.sqlite3"),
        )
    )
    llm_provider: str = os.getenv("BIZCHECK_LLM_PROVIDER", "disabled")
    openai_api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("BIZCHECK_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        ),
        repr=False,
    )
    openai_base_url: str = os.getenv(
        "BIZCHECK_OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    )
    openai_model: str = os.getenv("BIZCHECK_OPENAI_MODEL", "gpt-5.4")
    thu_api_key: str | None = field(
        default_factory=lambda: os.getenv("BIZCHECK_THU_API_KEY") or None,
        repr=False,
    )
    thu_base_url: str = os.getenv(
        "BIZCHECK_THU_BASE_URL",
        "https://api.ithu.tw/v1",
    )
    thu_model: str = os.getenv("BIZCHECK_THU_MODEL", "gpt-oss-120b")
    thu_json_mode: str = os.getenv("BIZCHECK_THU_JSON_MODE", "auto")
    thu_temperature: float = float(
        os.getenv("BIZCHECK_THU_TEMPERATURE", "0.1")
    )
    llm_timeout_seconds: float = float(
        os.getenv("BIZCHECK_LLM_TIMEOUT_SECONDS", "45")
    )
    llm_max_output_tokens: int = int(
        os.getenv("BIZCHECK_LLM_MAX_OUTPUT_TOKENS", "3000")
    )
    llm_max_attempts: int = int(
        os.getenv("BIZCHECK_LLM_MAX_ATTEMPTS", "2")
    )
    llm_comparison_cache_max_entries: int = int(
        os.getenv("BIZCHECK_LLM_COMPARISON_CACHE_MAX_ENTRIES", "128")
    )
    llm_comparison_cache_ttl_seconds: int = int(
        os.getenv("BIZCHECK_LLM_COMPARISON_CACHE_TTL_SECONDS", "900")
    )
    llm_comparison_cache_stale_if_error_seconds: int = int(
        os.getenv(
            "BIZCHECK_LLM_COMPARISON_CACHE_STALE_IF_ERROR_SECONDS",
            "3600",
        )
    )

    def __post_init__(self) -> None:
        environment = self.environment.strip().lower()
        normalized_origins = parse_cors_origins(",".join(self.cors_origins))
        if environment == "production":
            if not normalized_origins:
                raise ValueError(
                    "Production requires at least one BIZCHECK_CORS_ORIGINS value."
                )
            insecure_origins = tuple(
                origin for origin in normalized_origins if not origin.startswith("https://")
            )
            if insecure_origins:
                raise ValueError(
                    "Production CORS origins must use https: "
                    + ", ".join(insecure_origins)
                )
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "cors_origins", normalized_origins)


@lru_cache
def get_settings() -> Settings:
    return Settings()
