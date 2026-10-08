"""Probe HTML/assets, API predictions and the exact code/model/image identity."""

import re
import urllib.request
from urllib.parse import urlsplit

from fleetguard.serving.probe import probe, request_json


def probe_candidate(url, candidate):
    parsed = urlsplit(url)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    ):
        raise ValueError("Deployment probe requires HTTPS (HTTP only for localhost tests).")
    result = probe(url, wait_seconds=10)
    if result["model"] != candidate["model"]:
        raise ValueError("Served model differs from the approved candidate.")
    _, deployed, _ = request_json(url.rstrip("/") + "/v1/deployment")
    expected = {
        "git_sha": candidate["git_sha"],
        "bundle_sha256": candidate["bundle_sha256"],
        "image_digest": candidate["image"].split("@", 1)[1],
    }
    if deployed["deployment"] != expected or deployed["model"] != candidate["model"]:
        raise ValueError("Served deployment identity differs from the approved candidate.")
    with urllib.request.urlopen(url.rstrip("/") + "/", timeout=10) as response:
        if "text/html" not in response.headers.get("Content-Type", ""):
            raise ValueError("Web root is not HTML.")
        page = response.read(1024 * 1024).decode("utf-8")
    assets = re.findall(r'(?:src|href)="(/assets/[^"?#]+)"', page)
    if not assets or not any(asset.endswith(".js") for asset in assets):
        raise ValueError("Built frontend assets are missing.")
    for asset in assets:
        with urllib.request.urlopen(url.rstrip("/") + asset, timeout=10) as response:
            if response.status != 200 or "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Frontend asset did not load.")
    return {
        "inference_contract": True,
        "deployment_identity": True,
        "web_assets": True,
        "artificial_probe_rows": result["tested_rows"],
        "scope": "Deployment contract check, not official-test evaluation or a capacity benchmark",
    }
