import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from fuzzy_core import RayleighFuzzy


def test_frontend_roots_and_updates_before_release():
    node = os.environ.get("RAYLEIGH_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is only needed to run the frontend unit test.")
    references = []
    for sigma in [1e-6, .5, 1, 2, 1e6]:
        for alpha in [.001, .1, .5, .95, 1]:
            left, right = RayleighFuzzy(sigma).alpha_cut(alpha)
            references.append(dict(sigma=sigma, alpha=alpha, left=left, right=right))
    test = Path(__file__).with_suffix(".mjs")
    subprocess.run([node, str(test)], input=json.dumps(references), text=True, check=True)


def test_cluster_threshold_preview_and_commit():
    node = os.environ.get("RAYLEIGH_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is only needed to run the frontend unit test.")
    subprocess.run([node, str(Path(__file__).with_name("test_live_clusters.mjs"))], check=True)
