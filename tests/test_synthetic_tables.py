"""The synthetic run's paper tables use the same code and columns as the real-text tables."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from drybench.cli import main as cli
from drybench.report import main as report_main
from drybench.report import synthetic_tables

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_run(run: Path) -> None:
    """Items, two prompt variants, mock signals, two readers, reader check and filler check."""
    items, resp = run / "items.jsonl", run / "responses.jsonl"
    cli(["items", "--out", str(items), "--n-per-cell", "1"])
    cli(["generate", "--items", str(items), "--out", str(resp), "--backend", "mock", "--model-name", "m1",
         "--variant", "default", "concise"])
    gold = {json.loads(l)["item_id"]: json.loads(l)["gold_minimal"] for l in items.read_text().splitlines()}
    rows = [json.loads(l) for l in resp.read_text().splitlines()]
    for r in rows:
        filler = "Great question! " if r["variant"] == "default" else ""
        r["response"] = f"{filler}{gold[r['item_id']]} I hope this helps!"
    resp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cli(["signals", "--items", str(items), "--responses", str(resp), "--out", str(run / "signals.jsonl"),
         "--backend", "mock", "--model-name", "m1"])
    (run / "reader2_mock").mkdir()
    for sub in (run, run / "reader2_mock"):
        cli(["score", "--items", str(items), "--responses", str(resp), "--out", str(sub / "scored.jsonl"),
             "--reader-backend", "mock", "--mode", "loo", "--ablation-abs-tol", "2.0"])
        cli(["check-reader", "--items", str(items), "--reader-backend", "mock", "--n", "5",
             "--out", str(sub / "reader_check.json")])
    run_real = _load_script("run_real")
    run_real.filler_check(argparse.Namespace(filler_check_n=1, backend="mock"), "unused", run / "filler_check", "auto")


def test_synthetic_tables_match_the_real_columns(tmp_path):
    _synthetic_run(tmp_path)
    text = synthetic_tables(tmp_path)
    for head in ("## T1.", "## T2.", "## T3.", "## T4.", "## T5.", "## T6.", "## T7."):
        assert head in text, head
    assert text.index("## T2.") < text.index("## T3.") < text.index("## T4.") < text.index("## T5.")
    assert "prefix ablation" in text and "filler map (reader main)" in text  # T1: ablation row, baselines
    assert "complete answers" in text and "removable, complete answers" in text  # T2 as in the real tables
    t3 = text[text.index("## T3."): text.index("## T4.")]
    assert "Same, reader mock:" in t3  # 'Be concise.' for both readers
    assert "filler or not" in text and "wrong statement or not" in text  # T4 span agreement
    assert "Facts delivered by item design" in text
    assert "DRY" not in text and "v0.7.1" not in text
    report_main([str(tmp_path), "--quiet"])
    assert (tmp_path / "tables_synthetic.md").read_text(encoding="utf-8") == text


def test_paper_tables_cell_runs_on_a_synthetic_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _synthetic_run(run)
    make_colab = _load_script("make_colab")
    cell = make_colab.build_cell(f'RUN = "{run}"\nN_ANNOTATE = 2\n', template="paper_tables_template.py")
    assert "DRY" not in cell.split("PKG_B64 = ")[0]
    (tmp_path / "cell.py").write_text(cell)
    out = subprocess.run([sys.executable, tmp_path / "cell.py"], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "## T7." in out.stdout and "Annotation sample:" in out.stdout
    paper = run / "paper"
    assert (paper / "tables.md").exists() and (paper / "results_main.csv").exists()
    assert (paper / "results_mock.csv").exists() and (paper / "annotation" / "key.jsonl").exists()
    assert (run / "run_paper.zip").exists()


def test_span_origin_and_entropy_split():
    import pandas as pd

    from drybench.report import _entropy, span_origin

    doc = "The Eiffel Tower was completed in 1889 for the World's Fair. It is located in Paris near the Seine."
    assert span_origin("The tower was completed in 1889 for the World's Fair.", doc) == "copied"
    assert span_origin("This landmark remains a beloved global symbol of romance.", doc) == "own words"
    assert span_origin("Great question!", doc) == "short"
    rows = []
    for i, (kind, text, h) in enumerate([("kept", "The Eiffel Tower was completed in 1889.", 0.1),
                                         ("uninformative", "It is located in Paris near the Seine.", 0.2),
                                         ("uninformative", "This landmark remains a beloved global symbol.", 0.9),
                                         ("kept", "Its completion year answers the question asked.", 0.3)]):
        rows.append({"group": "m | default", "item_id": "x", "kind": kind, "text": text, "h_span": h,
                     "tokens": 5 + i})
    t = _entropy(pd.DataFrame(rows), {"x": doc})
    r = t.loc["m | default"]
    assert r["filler copied from the document"] == 0.5
    assert r["AUC, copied"] == 1.0 and r["AUC, own words"] == 1.0
    assert "AUC, copied" not in _entropy(pd.DataFrame(rows)).columns


def test_kappa_counts_a_contradicted_fact_as_zero():
    from drybench.analysis import build_tables
    from drybench.items import generate_items
    from drybench.filler import completeness

    assert completeness([13.8, -11.3], 30.0) == 13.8 / 30.0
    assert completeness([1.0], 0.0) != completeness([1.0], 0.0)  # NaN without gold information
    it = generate_items(n_per_cell=1, seed=0)[0]
    row = {"model": "m", "variant": "default", "item_id": it.item_id, "W": 0.0, "W_filler": 0.0,
           "W_misleading": 0.0, "tokens": 20, "tokens_kept": 20, "tokens_gold": 10, "spans": [],
           "per_probe": [13.8, -11.3], "info": 2.5, "info_gold": 27.6, "kappa": 2.5 / 27.6,
           "overhead": 20 / (2.5 / 27.6 * 10), "status": "ok", "facts": 0.5}
    items_df, _ = build_tables([it], [row], [])
    r = items_df.iloc[0]
    assert abs(r["kappa"] - 0.5) < 1e-9 and abs(r["overhead"] - 4.0) < 1e-9  # scores stored before v0.7.4
    items_df, _ = build_tables([it], [dict(row, status="reader_fail")], [])
    assert items_df.iloc[0]["kappa"] == row["kappa"]  # left as stored (NaN in real scores)


def test_missing_fact_items_are_judged_by_the_rule_not_the_filler_map(tmp_path):
    _synthetic_run(tmp_path)  # every answer is the gold answer, which says the missing fact is not stated
    text = synthetic_tables(tmp_path)
    t5 = text[text.index("## T5."): text.index("## T6.")]
    removable = t5[: t5.index("## T5. Filler")]
    assert "missing fact" not in removable  # left out of the filler-map tables
    missing = t5[t5.index("## T5. Missing-fact items"):]
    assert "| model \\| prompt | missing fact, k=1 | missing fact, k=3 | all | n |" in missing
    assert "| m1 \\| default | 1.000 | 1.000 | 1.000 | 2 |" in missing
    t2 = text[text.index("## T2."): text.index("## T3.")]
    assert "| m1 \\| default | 13 |" in t2  # 15 items, the 2 with a missing fact left out


def test_check_abstain_script(tmp_path, capsys):
    _synthetic_run(tmp_path)  # gold answers: every missing-fact answer says the fact is not given
    _load_script("check_abstain").main([str(tmp_path), "--labels", str(tmp_path / "none.json")])
    out = capsys.readouterr().out
    assert "4 missing-fact answers; the rule says the fact is not given in 1.000" in out
