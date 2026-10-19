"""v1.2.5: the confirmation run on a server with local model folders (scripts/run_final.py, the notebook)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IDS = ["meta-llama/Llama-3.2-3B-Instruct", "microsoft/Phi-4-mini-instruct", "Qwen/Qwen2.5-14B-Instruct",
       "Qwen/Qwen2.5-3B-Instruct"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_local_models_and_datasets(tmp_path, monkeypatch):
    rf = _load(ROOT / "scripts" / "run_final.py", "run_final")
    m = tmp_path / "models" / "Qwen2.5-3B-Instruct"
    m.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no config.json"):
        rf.preflight([str(m)])
    with pytest.raises(SystemExit, match="no config.json"):
        rf.preflight(["~/no/such/folder"])  # a path, not a hub id
    (m / "config.json").write_text("{}")
    rf.preflight([str(m)])  # a local folder: no hub call
    monkeypatch.setattr(rf.rr, "ENV", {})
    assert not rf.offline([str(m), "Qwen/Qwen2.5-3B-Instruct"]) and "HF_HUB_OFFLINE" not in rf.rr.ENV
    assert rf.offline([str(m)]) and rf.rr.ENV["HF_HUB_OFFLINE"] == "1"
    data = tmp_path / "data"
    with pytest.raises(SystemExit, match="Datasets not found"):
        rf.check_data(data, ["squad2"], [], need_all=True)
    rf.check_data(data, ["squad2"], [], need_all=False)  # online: run_real downloads it
    (data / "squad2").mkdir(parents=True)
    (data / "squad2" / "dev-v2.0.json").write_text("x")
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "items.jsonl").write_text("")
    recorded = "/content/drive/MyDrive/drybench_runs/data/squad2/dev-v2.0.json"
    (dev / "config.json").write_text(json.dumps({"data": {recorded: "0" * 64}}))
    with pytest.raises(SystemExit, match="sha256 differs"):
        rf.check_data(data, ["squad2"], [str(dev / "items.jsonl")], need_all=True)
    digest = rf.rr.sha256(data / "squad2" / "dev-v2.0.json")
    (dev / "config.json").write_text(json.dumps({"data": {recorded: digest, "/x/data/poddar/data.jsonl": "1" * 64}}))
    rf.check_data(data, ["squad2"], [str(dev / "items.jsonl")], need_all=True)  # Poddar et al. is not a suite


def test_free_model_cache_never_touches_a_local_folder(tmp_path, monkeypatch):
    import sys
    import types

    rr = _load(ROOT / "scripts" / "run_real.py", "run_real")
    called = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(scan_cache_dir=lambda: called.append(1)))
    rr.free_model_cache(str(tmp_path), set())
    assert not called


def test_final_cell_on_a_server(tmp_path, monkeypatch, capsys):
    """Every model a local folder (LOCAL): no hub, nothing installed, the development items in RUN_ROOT, the main
    reader in fp16 and the rest in bf16."""
    mc = _load(ROOT / "scripts" / "make_colab.py", "make_colab")
    root = tmp_path / "runs"
    local = {}
    for mid in IDS:
        (tmp_path / "models" / mid.split("/")[1]).mkdir(parents=True)
        local[mid] = str(tmp_path / "models" / mid.split("/")[1])
    cell = mc.build_cell(template="final_template.py")
    a, b = "# ---- settings ----\n", "# ---- end settings ----\n"
    settings = cell[cell.index(a) + len(a):cell.index(b)]
    assert "LOCAL = {}" in settings and 'RUN_ROOT = "~/drybench_runs"' in settings
    settings = settings.replace('RUN_ROOT = "~/drybench_runs"', f'RUN_ROOT = "{root}"').replace(
        "LOCAL = {}", f"LOCAL = {local!r}")
    cell = mc.build_cell(settings, template="final_template.py")
    ns = {"__name__": "cell"}
    monkeypatch.chdir(tmp_path)
    exec(compile(cell[:cell.rindex("\nmain()")], "cell", "exec"), ns)
    calls = []
    ns["run"] = lambda args: calls.append([str(x) for x in args])
    ns["main"]()
    printed = capsys.readouterr().out
    assert "Every model is a local folder" in printed and "nothing installed" in printed
    (cmd,) = calls
    assert cmd[1].endswith("scripts/run_final.py")

    def values(flag):
        i = cmd.index(flag) + 1
        out = []
        while i < len(cmd) and not cmd[i].startswith("--"):
            out.append(cmd[i])
            i += 1
        return out

    assert values("--reader") == [local["Qwen/Qwen2.5-3B-Instruct"]]
    assert values("--models") == [f"llama3.2-3b={local[IDS[0]]}", f"phi4-mini={local[IDS[1]]}",
                                  f"qwen2.5-14b={local[IDS[2]]}"]
    assert values("--judges") == [f"qwen2.5-14b={local[IDS[2]]}"]
    assert values("--reader-dtype") == ["fp16"] and values("--dtype") == ["bf16"]
    assert values("--real-check-n") == ["8"] and values("--reader-items-per-cell") == ["qwen2.5-14b=10"]
    dev = values("--dev-items")  # both development items files come with the package
    assert len(dev) == 2 and dev[0].endswith("results/dev_v1.0/real/items.jsonl")
    assert dev[1].endswith("results/dev_v1.2/real_t4_n15_v12/items.jsonl") and all(Path(d).exists() for d in dev)
    assert Path(dev[1]).with_name("config.json").exists()  # its datasets' sha256, checked by run_final
    assert values("--data") == [str(root / "data")] and "--free-model-cache" not in cmd


def test_compare_reader_on_a_mock_run(tmp_path):
    """The mock reader scores a run's responses as the run did."""
    tb = _load(Path(__file__).with_name("test_baselines_v1.py"), "tb")
    run = tmp_path / "syn"
    run.mkdir()
    tb._synthetic_run(run)
    cr = _load(ROOT / "scripts" / "compare_reader.py", "compare_reader")
    out = tmp_path / "eq"
    cr.main([str(run), "--reader", "mock", "--backend", "mock", "--dtype", "fp32", "--n", "5", "--out", str(out)])
    s = json.loads((out / "summary.json").read_text())["runs"]["syn"]["fp32"]
    assert s["responses"] == 5 and s["same_status"] == 5 and s["same_kept"] == s["with_map_in_both"]
    assert s["max_d_info"] == pytest.approx(0.0, abs=1e-9) and "| syn | fp32 | 5 |" in (out / "summary.md").read_text()
    cr.main([str(run), "--reader", "mock", "--backend", "mock", "--dtype", "fp32", "--n", "5", "--out", str(out)])
    assert len((out / "syn" / "scored_fp32.jsonl").read_text().splitlines()) == 5  # a rerun reuses the scores
