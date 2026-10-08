from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from fleetguard.release_smoke import create_fixture  # noqa: E402
from fleetguard.serving.api import create_app  # noqa: E402
from fleetguard.serving.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def web_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("web-api")
    _, _, release = create_fixture(root)
    web = root / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>FleetGuard UI</title>", encoding="utf-8")
    (web / "assets").mkdir()
    (web / "assets/app.js").write_text("console.log('UI')", encoding="utf-8")
    return Settings(release_dir=release, web_dir=web, allow_synthetic=True)


def test_optional_static_ui_preserves_api_and_readiness(web_settings):
    with TestClient(create_app(web_settings)) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "FleetGuard UI" in response.text
        assert response.headers["x-request-id"]
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/health/ready").json()["status"] == "ready"
        info = client.get("/v1/model").json()
        deployment = client.get("/v1/deployment").json()
        assert deployment["model"] == info["model"]
        assert deployment["deployment"] == {
            "git_sha": None,
            "bundle_sha256": None,
            "image_digest": None,
        }
        row = dict.fromkeys(info["feature_names"])
        assert client.post("/v1/predict", json={"sensors": row}).status_code == 200
        assert client.get("/docs").status_code == 200
        assert "/v1/predict" in client.get("/openapi.json").json()["paths"]


def test_static_server_does_not_expose_release_or_parent_files(web_settings):
    (web_settings.web_dir.parent / "private.txt").write_text("private", encoding="utf-8")
    with TestClient(create_app(web_settings)) as client:
        for path in ["/run.json", "/private.txt", "/%2e%2e/private.txt", "/assets/missing.js"]:
            response = client.get(path)
            assert response.status_code == 404
            assert (
                "private" not in response.text or response.json()["error"]["code"] == "http_error"
            )


def test_build_directory_is_explicit_and_must_be_complete(web_settings, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="built frontend"):
        create_app(replace(web_settings, web_dir=tmp_path))
    monkeypatch.setenv("FLEETGUARD_WEB_DIR", str(web_settings.web_dir))
    assert Settings.from_env(web_settings.release_dir).web_dir == web_settings.web_dir.resolve()
    with TestClient(create_app(replace(web_settings, web_dir=None))) as client:
        assert client.get("/").status_code == 404
