"""Non-secret code/model provenance exposed for deployment checks."""

import os
import re


def deployment_identity():
    values = {
        "git_sha": os.environ.get("FLEETGUARD_GIT_SHA"),
        "bundle_sha256": os.environ.get("FLEETGUARD_BUNDLE_SHA256"),
        "image_digest": os.environ.get("FLEETGUARD_IMAGE_DIGEST"),
    }
    for key, pattern in (
        ("git_sha", r"[0-9a-f]{40}"),
        ("bundle_sha256", r"[0-9a-f]{64}"),
        ("image_digest", r"sha256:[0-9a-f]{64}"),
    ):
        if values[key] is not None and not re.fullmatch(pattern, values[key]):
            raise ValueError(f"Invalid deployment provenance: {key}.")
    return values
