from __future__ import annotations

import httpx
import pytest

from app.main import create_app
from app.settings import DEFAULT_CORS_ORIGINS, Settings, parse_cors_origins


pytestmark = pytest.mark.anyio


def test_development_defaults_cover_localhost_and_loopback_vite() -> None:
    assert "http://localhost:5173" in DEFAULT_CORS_ORIGINS
    assert "http://127.0.0.1:5173" in DEFAULT_CORS_ORIGINS


def test_cors_parser_normalizes_deduplicates_and_preserves_exact_origins() -> None:
    origins = parse_cors_origins(
        " https://app.bizcheck.example:443/, https://app.bizcheck.example,"
        "https://admin.bizcheck.example "
    )

    assert origins == (
        "https://app.bizcheck.example",
        "https://admin.bizcheck.example",
    )


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "https://*.bizcheck.example",
        "ftp://app.bizcheck.example",
        "https://user:secret@app.bizcheck.example",
        "https://app.bizcheck.example/api",
        "https://app.bizcheck.example?preview=1",
        "https://app.bizcheck.example/#fragment",
        "https://app.bizcheck.example:invalid",
        "https://app bizcheck.example",
        "https://app.bizcheck.example\\evil",
    ],
)
def test_cors_parser_rejects_wildcard_or_non_origin_values(origin: str) -> None:
    with pytest.raises(ValueError):
        parse_cors_origins(origin)


def test_production_requires_explicit_https_origin() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Settings(environment="production", cors_origins=())

    with pytest.raises(ValueError, match="must use https"):
        Settings(
            environment="production",
            cors_origins=("http://app.bizcheck.example",),
        )


def test_settings_reads_deployment_origin_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "BIZCHECK_CORS_ORIGINS",
        "https://app.bizcheck.example,https://www.bizcheck.example",
    )

    settings = Settings(environment="production")

    assert settings.cors_origins == (
        "https://app.bizcheck.example",
        "https://www.bizcheck.example",
    )


async def test_deployed_origin_get_and_post_preflights_are_allowed() -> None:
    origin = "https://app.bizcheck.example"
    app = create_app(Settings(environment="production", cors_origins=(origin,)))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        for method, path in (
            ("GET", "/api/v1/companies/search?q=宏碁"),
            ("POST", "/api/v1/companies/compare"),
        ):
            response = await client.options(
                path,
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": "content-type",
                },
            )

            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin
            assert method in response.headers["access-control-allow-methods"]
            assert response.headers["access-control-max-age"] == "600"
            assert "Origin" in response.headers["vary"]


async def test_unlisted_origin_is_not_granted_cors_access() -> None:
    allowed_origin = "https://app.bizcheck.example"
    rejected_origin = "https://attacker.example"
    app = create_app(
        Settings(environment="production", cors_origins=(allowed_origin,))
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        preflight = await client.options(
            "/api/v1/companies/compare",
            headers={
                "Origin": rejected_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        simple = await client.get(
            "/health/live",
            headers={"Origin": rejected_origin},
        )

    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers
    assert simple.status_code == 200
    assert "access-control-allow-origin" not in simple.headers
