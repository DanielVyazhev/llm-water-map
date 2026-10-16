"""v1.2: the Colab notebooks embed the current code (byte for byte), and what each one needs."""

import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_colab():
    spec = importlib.util.spec_from_file_location("make_colab", ROOT / "scripts" / "make_colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _blob(cell: str) -> str:
    return cell.split('PKG_B64 = "', 1)[1].split('"', 1)[0]


def _payload(cell: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(_blob(cell))))


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
        "probe_models.ipynb": json.dumps(mod.probe_notebook(mod.build_cell(template="probe_template.py")), indent=1),
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


def test_notebooks_hide_the_package():
    """v1.2.1: the package is the first code cell, its code hidden, and no other cell carries it; a cell that needs
    it stops with a message when the cell "Package" has not run."""
    mod = _make_colab()
    builders = {"cell_template.py": mod.notebook, "real_template.py": mod.real_notebook,
                "missing_check_template.py": mod.missing_notebook, "final_template.py": mod.final_notebook,
                "dev_baselines_template.py": mod.dev_baselines_notebook, "probe_template.py": mod.probe_notebook}
    for template, build in builders.items():
        cell = mod.build_cell(template=template)
        codes = [c for c in build(cell)["cells"] if c["cell_type"] == "code"]
        sources = ["".join(c["source"]) for c in codes]
        assert [i for i, s in enumerate(sources) if 'PKG_B64 = "' in s] == [0], template
        assert codes[0]["metadata"]["cellView"] == "form" and sources[0].startswith("#@title Package v")
        assert _blob(sources[0]) == _blob(cell) and mod.split_package(cell)[1] in sources, template
        assert not any(c["metadata"] for c in codes[1:]), template  # the other cells show their code
        needs = [s for s in sources[1:] if any(f"\ndef {f}():" in s for f in ("install", "load_package", "extract"))]
        assert len(needs) == (2 if template == "cell_template.py" else 1), template  # + the paper tables cell
        for src in needs:
            ns = {"__name__": "colab_cell"}
            exec(src.rstrip().rsplit("\n", 1)[0], ns)  # the definitions, without the last line (the run)
            with pytest.raises(SystemExit, match='Run the cell "Package" first'):
                (ns.get("install") or ns.get("load_package") or ns["extract"])()


def test_dev_cell_reads_local_models_without_the_hub(tmp_path):
    """v1.2.2: on a server whose models are local folders, nothing is fetched from the Hugging Face hub."""
    cell = _make_colab().build_cell(template="dev_baselines_template.py")
    ns = {"__name__": "colab_cell"}
    exec(cell.split('\nPKG_B64 = "')[0], ns)  # settings and definitions; no package, no run
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        ns["ENV"].pop(key, None)
    (tmp_path / "judge").mkdir()
    (tmp_path / "small").mkdir()
    ns["JUDGES"], ns["SELECTIVE"] = {"j": str(tmp_path / "judge")}, {"s": str(tmp_path / "small")}
    ns["local_models"]()
    assert ns["ENV"]["HF_HUB_OFFLINE"] == "1" and ns["ENV"]["TRANSFORMERS_OFFLINE"] == "1"
    ns["ENV"].pop("HF_HUB_OFFLINE")
    ns["SELECTIVE"] = {"s": "Qwen/Qwen2.5-3B-Instruct"}  # a hub id: the hub stays on
    ns["local_models"]()
    assert "HF_HUB_OFFLINE" not in ns["ENV"]
