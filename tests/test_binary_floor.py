"""v0.8.1: a missing fact asked as a yes/no question has a floor of 0.1 (a made-up value no longer
'delivers' it), and a new scoring version rescores only the answers it changed."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

from drybench.backends.mock import MockBackend
from drybench.cli import main as cli
from drybench.items import generate_items
from drybench.reader import BINARY_FLOOR, BINARY_STYLES, MISSING_STYLES, Reader, ReaderConfig, is_missing
from drybench.rescore import plan, prune
from drybench.segment import segment
from drybench.filler import dry_score

ROOT = Path(__file__).resolve().parents[1]


def _item():
    return next(it for it in generate_items(n_per_cell=1, seed=0) if not it.knobs["answerable"] and it.knobs["k"] > 1)


def _with(item, sentence):
    """The gold answer with its 'not stated' sentence replaced (None: left out)."""
    parts = [s.text for s in segment(item.gold_minimal)]
    return " ".join(sentence if "not stated" in p else p for p in parts if sentence or "not stated" not in p)


def _chat(**kw):
    return Reader(MockBackend(), ReaderConfig(use_chat=True, **kw))


def test_floor_is_for_missing_facts_asked_yes_or_no():
    it = _item()
    miss = next(p for p in it.required_probes if is_missing(p))
    other = next(p for p in it.required_probes if not is_missing(p))
    for style in MISSING_STYLES:
        rd = _chat(missing_style=style)
        assert rd.floor(miss) == (BINARY_FLOOR if style in BINARY_STYLES else 1e-6), style
        assert rd.floor(other) == 1e-6
    assert Reader(MockBackend(), ReaderConfig()).floor(miss) == 1e-6  # no chat template: no yes/no question
    assert _chat().log_floors([miss, other]).tolist() == [math.log(0.1), math.log(1e-6)]


def test_made_up_value_for_a_missing_fact_is_deleted_as_wrong():
    it = _item()
    j = next(i for i, p in enumerate(it.required_probes) if is_missing(p))
    resp = _with(it, "The cargo was WRONG bananas.")
    old = dry_score(_chat(binary_floor=1e-6), it, resp, mode="loo")  # v0.8.0: p(yes) = 0.01 is 8.5 nats
    new = dry_score(_chat(), it, resp, mode="loo")
    kind = lambda res: next(s.kind for s in res.spans if "WRONG" in s.text)  # noqa: E731
    assert old.per_probe[j] > 8 and kind(old) == "kept"
    assert new.per_probe[j] == 0.0 and kind(new) == "misleading" and new.W_misleading > 0
    assert new.facts == old.facts < 1.0  # p-based: the same either way
    # the missing fact weighs ln(0.8 / 0.1) nats in kappa now (a yes/no fact), the others ~9 each
    assert new.kappa == pytest.approx((new.per_probe[0] + new.per_probe[2]) / new.info_gold)


def test_abstention_is_still_kept_and_counted():
    it = _item()
    j = next(i for i, p in enumerate(it.required_probes) if is_missing(p))
    res = dry_score(_chat(), it, "Great question! " + it.gold_minimal, mode="loo")
    kinds = {s.text: s.kind for s in res.spans}
    assert kinds["Great question!"] == "uninformative"
    assert all(k == "kept" for t, k in kinds.items() if "not stated" in t)
    assert res.per_probe[j] == pytest.approx(math.log(0.8 / BINARY_FLOOR))  # mock: p(yes) = 0.8
    assert res.facts == 1.0 and res.kappa == pytest.approx(1.0)
    silent = dry_score(_chat(), it, _with(it, None), mode="loo")  # says nothing about the cargo
    assert silent.per_probe[j] == 0.0 and silent.facts < 1.0


def test_rescore_plan():
    qwen, phi = "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct"
    assert plan("7", "8", qwen) == "missing" and plan("7", "8", phi) == "none"
    assert plan("8", "8", qwen) == "none"
    assert plan("6", "8", qwen) == "all" and plan("older", "8", phi) == "all" and plan(None, "8", qwen) == "all"
    assert plan("7", "8", None) == "all"  # a folder whose reader is unknown


def _folder(tmp_path):
    items = generate_items(n_per_cell=1, seed=0)
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(it.to_dict()) + "\n" for it in items))
    missing = {it.item_id for it in items if any(is_missing(p) for p in it.required_probes)}
    rows = [{"item_id": it.item_id, "model": "m"} for it in items]
    (tmp_path / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for d in ("filler_check_real", "filler_check", "human_check", "report"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "summary.json").write_text("{}")
    (tmp_path / "filler_check_real" / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "reader_check.json").write_text("{}")
    return items, missing


def test_prune_keeps_what_a_partial_change_does_not_touch(tmp_path):
    items, missing = _folder(tmp_path)
    assert missing and len(missing) < len(items)
    log = prune(tmp_path, tmp_path / "items.jsonl", "missing")
    left = [json.loads(x)["item_id"] for x in (tmp_path / "scored.jsonl").read_text().splitlines()]
    assert set(left) == {it.item_id for it in items} - missing
    assert [json.loads(x)["item_id"] for x in (tmp_path / "filler_check_real" / "rows.jsonl").read_text().splitlines()] == left
    assert not (tmp_path / "reader_check.json").exists() and not (tmp_path / "report").exists()
    assert not (tmp_path / "filler_check_real" / "summary.json").exists()
    assert (tmp_path / "filler_check" / "summary.json").exists() and (tmp_path / "human_check" / "summary.json").exists()
    assert any("rows of items with a missing fact removed" in line for line in log)
    assert prune(tmp_path, tmp_path / "items.jsonl", "none") == []
    prune(tmp_path, tmp_path / "items.jsonl", "all")
    assert not any((tmp_path / n).exists() for n in ("scored.jsonl", "filler_check", "human_check", "filler_check_real"))


def test_prune_scores_cli(tmp_path, capsys):
    _folder(tmp_path)
    cli(["prune-scores", "--dir", str(tmp_path), "--items", str(tmp_path / "items.jsonl"),
         "--reader", "microsoft/Phi-3.5-mini-instruct", "--from", "7", "--to", "8", "--json"])
    assert json.loads(capsys.readouterr().out) == {"plan": "none", "changes": []}
    cli(["prune-scores", "--dir", str(tmp_path), "--items", str(tmp_path / "items.jsonl"),
         "--reader", "Qwen/Qwen2.5-3B-Instruct", "--from", "7", "--to", "8"])
    assert "rescore missing" in capsys.readouterr().out


def test_run_real_rescores_per_reader(tmp_path):
    spec = importlib.util.spec_from_file_location("run_real", ROOT / "scripts" / "run_real.py")
    run_real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_real)
    import drybench

    items, missing = _folder(tmp_path)
    r2 = tmp_path / "reader2_phi-3.5-mini"
    r2.mkdir()
    (r2 / "scored.jsonl").write_text((tmp_path / "scored.jsonl").read_text())
    (tmp_path / "versions.json").write_text(json.dumps({"scoring": str(int(drybench.SCORING_VERSION) - 1),
                                                        "signals": drybench.SIGNALS_VERSION}))
    (tmp_path / "filter_report.json").write_text(json.dumps({"scoring": str(int(drybench.SCORING_VERSION) - 1)}))
    run_real.check_versions(tmp_path, "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct")
    assert len((tmp_path / "scored.jsonl").read_text().splitlines()) == len(items) - len(missing)
    assert len((r2 / "scored.jsonl").read_text().splitlines()) == len(items)  # Phi asks it plainly: unchanged
    assert json.loads((tmp_path / "filter_report.json").read_text())["scoring"] == drybench.SCORING_VERSION
    assert json.loads((tmp_path / "versions.json").read_text())["scoring"] == drybench.SCORING_VERSION
