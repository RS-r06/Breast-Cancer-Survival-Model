"""The browser calculator must give the same answers as the Python model."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODEL = Path("docs") / "model.json"


@pytest.mark.skipif(shutil.which("node") is None or not MODEL.exists(), reason="needs Node and a fitted model")
def test_javascript_matches_python():
    script = """
    const m = require(process.argv[1] + '/docs/model.json');
    const { riskAt5 } = require(process.argv[1] + '/docs/model.js');
    console.log(JSON.stringify(m.examples.map(e => riskAt5(m, e.inputs))));
    """
    out = subprocess.run(["node", "-e", script, str(Path.cwd())], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    py = [e["risk_5y"] for e in json.loads(MODEL.read_text())["examples"]]
    assert js == pytest.approx(py, abs=1e-9)
    assert py[0] < py[1] < py[2]  # the three profiles run from low to high risk
