"""Tests for the SENPAI FastAPI app.

httpx (and therefore Starlette's TestClient) is not a project dependency, so
these tests drive the app at the factory and route-coroutine level rather than
over HTTP. All host-touching startup (Astrometry.net validation, process
pools) and the heavy processing layer are mocked, so the suite runs offline,
deterministically, and fast.

Covered:
* ``create_app`` construction + route registration + config.version wiring
* the ``lifespan`` startup/shutdown (executor + astrometry hooks)
* index/health handlers returning config.version
* request-validation failures (pydantic models reject bad payloads)
* happy paths with the collect pipeline + solve_field mocked
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from senpai.api.models.returns import DetectResponse, FrameResult
from senpai.api.routes.senpai import FilePayloadItem
from senpai.engine.models.metadata import CollectionMetadata
from senpai.engine.models.senpai import SenpaiRun

from .conftest import make_request

# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------


def test_create_app_uses_config_version(patched_app_env, _init_config):
    app = patched_app_env.create_app(_init_config)
    assert app.title == "SENPAI API"
    assert app.version == _init_config.version


def test_create_app_registers_expected_routes(patched_app_env, _init_config):
    app = patched_app_env.create_app(_init_config)
    # Assert against the OpenAPI path set (the stable public contract) rather than
    # introspecting app.routes: FastAPI 0.137 keeps lazy _IncludedRouter wrappers
    # in app.routes (no .path), so a {route.path ...} comprehension breaks there.
    paths = set(app.openapi()["paths"])
    assert "/senpai/" in paths
    assert "/senpai/detect" in paths
    assert "/senpai/detect/upload" in paths
    assert "/senpai/sidereal" in paths
    assert "/senpai/rate" in paths
    assert "/astrometry/" in paths
    assert "/astrometry/solve/sources" in paths


def test_create_app_openapi_schema(patched_app_env, _init_config):
    app = patched_app_env.create_app(_init_config)
    schema = app.openapi()
    assert schema["info"]["title"] == "SENPAI API"
    assert "/senpai/detect" in schema["paths"]


def test_create_app_accepts_config_path(patched_app_env):
    from senpai.core.constants import LOCAL_APP_CONFIG_OVERRIDE

    app = patched_app_env.create_app(LOCAL_APP_CONFIG_OVERRIDE)
    assert app.version is not None


# ---------------------------------------------------------------------------
# lifespan — executor + astrometry validation
# ---------------------------------------------------------------------------


def test_lifespan_runs_hermetically(patched_app_env, _init_config):
    """The lifespan must validate astrometry and set up/tear down the executor."""
    app = patched_app_env.create_app(_init_config)

    async def drive():
        async with patched_app_env.lifespan(app):
            assert app.state.executor is not None

    asyncio.run(drive())


# ---------------------------------------------------------------------------
# index / health handlers
# ---------------------------------------------------------------------------


def test_senpai_index_returns_version(_init_config):
    from senpai.api.routes.senpai import index

    result = asyncio.run(index(make_request("/senpai/")))
    assert result["version"] == _init_config.version
    assert "api" in result


def test_astrometry_index_returns_config(_init_config):
    from senpai.api.routes.astrometry import index

    response = asyncio.run(index(make_request("/astrometry/")))
    body = json.loads(response.body)
    assert "api" in body
    assert body["config"]["version"] == _init_config.version


# ---------------------------------------------------------------------------
# Request-model validation (the pydantic layer FastAPI uses for 422s)
# ---------------------------------------------------------------------------


def test_file_payload_item_requires_file():
    with pytest.raises(ValidationError):
        FilePayloadItem()


def test_file_payload_item_rejects_wrong_type():
    with pytest.raises(ValidationError):
        FilePayloadItem(file=123)


def test_file_payload_item_optional_sequence_fields():
    item = FilePayloadItem(file="ZmFrZQ==")
    assert item.sequence_id is None
    assert item.sequence_count is None


# ---------------------------------------------------------------------------
# /senpai/detect, /sidereal, /rate — happy path with mocked pipeline
# ---------------------------------------------------------------------------


def _canned_run() -> SenpaiRun:
    """A minimal completed SenpaiRun with no frames (avoids heavy models)."""
    return SenpaiRun(
        id="test-run",
        num_frames=0,
        completed=True,
        collect_metadata=CollectionMetadata(),
    )


@pytest.fixture
def mocked_pipeline(monkeypatch):
    """Patch the collect route's I/O and processing to canned outputs."""
    import senpai.api.routes.senpai as route_mod

    calls = {"loaded": [], "processed": 0}

    def fake_load(encoded_files):
        calls["loaded"].append(list(encoded_files))
        return [object() for _ in encoded_files]

    def fake_process(file_list):
        calls["processed"] += 1
        return _canned_run()

    monkeypatch.setattr(route_mod, "load_base64_files", fake_load)
    monkeypatch.setattr(route_mod, "process_senpai_collect", fake_process)
    return calls


@pytest.mark.parametrize(
    "handler_name,path",
    [
        ("detect", "/senpai/detect"),
        ("process_sidereal", "/senpai/sidereal"),
        ("process_rate", "/senpai/rate"),
    ],
)
def test_collect_endpoints_happy_path(mocked_pipeline, handler_name, path):
    import senpai.api.routes.senpai as route_mod

    handler = getattr(route_mod, handler_name)
    payload = [FilePayloadItem(file="ZmFrZQ==", sequence_id=0, sequence_count=1)]
    result = asyncio.run(handler(make_request(path), payload))

    assert isinstance(result, DetectResponse)
    assert result.frames == []
    assert result.correlated_streaks == []
    assert mocked_pipeline["processed"] == 1
    assert mocked_pipeline["loaded"] == [["ZmFrZQ=="]]


def test_detect_upload_alias(mocked_pipeline):
    from senpai.api.routes.senpai import detect_upload

    payload = [FilePayloadItem(file="ZmFrZQ==")]
    result = asyncio.run(detect_upload(make_request("/senpai/detect/upload"), payload))
    assert isinstance(result, DetectResponse)
    assert mocked_pipeline["processed"] == 1


def test_detect_passes_all_files_to_loader(mocked_pipeline):
    from senpai.api.routes.senpai import detect

    payload = [FilePayloadItem(file=f) for f in ("QQ==", "Qg==", "Qw==")]
    asyncio.run(detect(make_request("/senpai/detect"), payload))
    assert mocked_pipeline["loaded"] == [["QQ==", "Qg==", "Qw=="]]


# ---------------------------------------------------------------------------
# /astrometry/solve/sources — happy path + status-code contract
# ---------------------------------------------------------------------------


def _starfield(fit: bool):
    from senpai.engine.models.starfield import ImageMetadata, StarField

    return StarField(
        detections=[],
        image_metadata=ImageMetadata(image_id="x", width=10, height=10),
        fit=fit,
        wcs=None,
    )


def _starlist_image():
    from senpai.engine.models.starfield import ImageMetadata, StarInImage, StarListImage

    return StarListImage(
        detections=[StarInImage(x=1.0, y=2.0, counts=100.0)],
        image_metadata=ImageMetadata(image_id="x", width=10, height=10),
    )


def test_solve_sources_solved_returns_200(monkeypatch):
    import senpai.api.routes.astrometry as astro_mod

    monkeypatch.setattr(astro_mod, "solve_field", lambda sources: _starfield(True))
    response = asyncio.run(astro_mod.solve_sources(make_request("/astrometry/solve/sources"), _starlist_image()))
    assert response.status_code == 200
    assert json.loads(response.body)["fit"] is True


def test_solve_sources_unsolved_returns_422(monkeypatch):
    import senpai.api.routes.astrometry as astro_mod

    monkeypatch.setattr(astro_mod, "solve_field", lambda sources: _starfield(False))
    response = asyncio.run(astro_mod.solve_sources(make_request("/astrometry/solve/sources"), _starlist_image()))
    assert response.status_code == 422


def test_solve_sources_passes_sources_through(monkeypatch):
    import senpai.api.routes.astrometry as astro_mod

    seen = {}

    def fake_solve(sources):
        seen["n"] = len(sources.detections)
        return _starfield(True)

    monkeypatch.setattr(astro_mod, "solve_field", fake_solve)
    asyncio.run(astro_mod.solve_sources(make_request("/astrometry/solve/sources"), _starlist_image()))
    assert seen["n"] == 1


# ---------------------------------------------------------------------------
# Response-model defaults sanity
# ---------------------------------------------------------------------------


def test_frame_result_defaults_are_json_serializable():
    fr = FrameResult(index=0)
    dumped = fr.model_dump(mode="json")
    assert dumped["index"] == 0
    assert dumped["detections"] == []
    assert dumped["astrometry"]["solved"] is False


def test_detect_response_roundtrip():
    resp = DetectResponse(frames=[FrameResult(index=2, tracking_mode="rate")])
    reparsed = DetectResponse(**resp.model_dump(mode="json"))
    assert reparsed.frames[0].index == 2
    assert reparsed.frames[0].tracking_mode == "rate"


# ---------------------------------------------------------------------------
# OpenAPI example payloads
# ---------------------------------------------------------------------------


def test_star_list_image_example_constructs():
    # examples.py used to pass ``stars=`` (a nonexistent field); the OpenAPI
    # example payload for /solve/sources must build a valid StarListImage.
    from senpai.api.models.examples import StarListImageExample

    example = StarListImageExample().value
    assert len(example.detections) > 0
    assert example.image_metadata.width == 1024
