"""Copy only the frozen inference contract; omit data, predictions and reports."""

import shutil

from fleetguard.experiment import require_complete
from fleetguard.io import read_json


def bundle_release(source, destination):
    state = require_complete(source)
    if state.get("protocol") != "release_v1":
        raise ValueError("Only an audited release can be bundled.")
    freeze = read_json(source / "freeze.json")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        # run.json is copied last, so incomplete bundles cannot become ready.
        for relative in [*freeze["files"], "freeze.json", "run.json"]:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, path)
            path.chmod(0o644)
        for path in destination.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        destination.chmod(0o755)
        require_complete(destination)
    except BaseException:
        # Only remove this newly-created destination; never touch the source.
        shutil.rmtree(destination)
        raise
    return destination
