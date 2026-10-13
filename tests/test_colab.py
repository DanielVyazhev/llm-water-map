"""v1.2: the Colab notebooks embed the current code (byte for byte), and what each one needs."""

import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _make_colab():
    spec = importlib.util.spec_from_file_location("make_colab", ROOT / "scripts" / "make_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload(cell: str) -> zipfile.ZipFile:
    blob = cell.split('PKG_B64 = "', 1)[1].split('"', 1)[0]
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(blob)))


def test_package_zip_is_reproducible():
    mod = _make_colab()
    assert mod.package_zip() == mod.package_zip()
    z = zipfile.ZipFile(io.BytesIO(mod.package_zip()))
    assert {i.date_time for i in z.infolist()} == {mod.ZIP_DATE}
    assert z.namelist() == sorted(z.namelist()) and "LICENSE" in z.namelist()


def test_notebooks_embed_the_current_code():
    """Fails after a code change until `python scripts/make_colab.py` rebuilds the notebooks."""
    mod = _make_colab()
    cell = mod.build_cell()
    real = mod.build_cell(template="real_template.py")
    final = mod.build_cell(template="final_template.py")
    dev = mod.build_cell(template="dev_baselines_template.py")
    expected = {
        "drybench_colab_cell.py": cell,
        "paper_tables_cell.py": mod.build_cell(template="paper_tables_template.py"),
        "real_texts_cell.py": real,
        "final_a100_cell.py": final,
        "drybench_colab.ipynb": json.dumps(mod.notebook(cell), indent=1),
        "real_texts_colab.ipynb": json.dumps(mod.real_notebook(real), indent=1),
        "missing_check_colab.ipynb": json.dumps(mod.missing_notebook(mod.build_cell(template="missing_check_template.py")),
                                                indent=1),
        "final_a100.ipynb": json.dumps(mod.final_notebook(final), indent=1),
        "dev_baselines_a100.ipynb": json.dumps(mod.dev_baselines_notebook(dev), indent=1),
    }
    stale = [f for f, text in expected.items() if (ROOT / "colab" / f).read_text() != text]
    assert not stale, f"rebuild with scripts/make_colab.py: {stale}"
    for f in ("drybench/confirm.py", "scripts/run_final.py", "docs/PROTOCOL.md"):  # same bytes as the repository
        assert _payload(final).read(f) == (ROOT / f).read_bytes(), f


def test_final_and_dev_notebooks_carry_what_they_need():
    mod = _make_colab()
    final = mod.build_cell(template="final_template.py")
    names = _payload(final).namelist()
    assert "results/dev_v1.0/real/items.jsonl" in names  # the v1.0 development items, excluded by the run
    assert 'pkg / "results" / "dev_v1.0" / "real" / "items.jsonl"' in final
    dev = mod.build_cell(template="dev_baselines_template.py")
    assert "__PKG_B64__" not in dev and "baselines" in dev and '"confirm"' in dev
    assert "results/dev_v1.0/real/items.jsonl" not in _payload(dev).namelist()
    nb = mod.dev_baselines_notebook(dev)
    assert nb["metadata"]["colab"]["gpuType"] == "A100" and "DRY" not in dev.split("PKG_B64 = ")[0]
