"""Build the self-contained Colab cell and notebook from colab/cell_template.py.

The drybench package (code + pyproject + README) is zipped, base64-encoded and embedded in
the cell, so the notebook needs no uploads.

  python scripts/make_colab.py            → colab/drybench_colab.ipynb, colab/drybench_colab_cell.py,
                                            real_texts_colab.ipynb, missing_check_colab.ipynb, final_a100.ipynb,
                                            dev_baselines_a100.ipynb, probe_models.ipynb

The embedded zip is byte-reproducible (sorted files, fixed dates and permissions): the same code gives the same
notebooks. In the notebooks the package line (PKG_B64) is a cell of its own, the first code cell, with its code
hidden, so the cells' output does not sit below it; the .py cells keep it at the end.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the version stamped on the notebooks is the package they embed


PACKAGED = ["LICENSE", "pyproject.toml", "README.md", "scripts/pilot_validation.py", "scripts/run_real.py",
            "scripts/run_synthetic.py", "scripts/run_final.py", "scripts/show_checks.py", "scripts/trace_response.py",
            "scripts/chat_annotators.py", "scripts/check_abstain.py", "scripts/probe_models.py",
            "scripts/compare_reader.py", "scripts/r3_losses.py",
            "annotation/dry_span_labels_offline.html", "docs/ANNOTATION.md", "docs/PROTOCOL.md",
            "results/abstain_labels.json"]
ZIP_DATE = (2020, 1, 1, 0, 0, 0)


# the confirmation run excludes the source texts of the development items: the notebook ships them (and the v1.2
# run's config.json, whose dataset sha256 the run checks)
EXTRA = {"final_template.py": ["results/dev_v1.0/real/items.jsonl", "results/dev_v1.2/real_t4_n15_v12/items.jsonl",
                               "results/dev_v1.2/real_t4_n15_v12/config.json"]}


def package_files(extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    return sorted(set(PACKAGED) | set(extra) | {p.relative_to(ROOT).as_posix() for p in (ROOT / "drybench").rglob("*.py")})


def package_zip(extra: tuple[str, ...] | list[str] = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in package_files(extra):
            info = zipfile.ZipInfo(f, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (ROOT / f).read_bytes())
    return buf.getvalue()


def build_cell(overrides: str | None = None, template: str = "cell_template.py") -> str:
    from drybench import __version__

    tmpl = (ROOT / "colab" / template).read_text()
    cell = tmpl.replace("__VERSION__", __version__).replace(
        "__PKG_B64__", base64.b64encode(package_zip(EXTRA.get(template, ()))).decode())
    if overrides is not None:  # replace the settings block (used for local dry runs)
        start, end = "# ---- settings ----\n", "# ---- end settings ----\n"
        a, b = cell.index(start) + len(start), cell.index(end)
        cell = cell[:a] + overrides + cell[b:]
    return cell


PKG_LINE = 'PKG_B64 = "'


def split_package(cell: str) -> tuple[str, str]:
    """The cell's package line and the cell without it (and without the blank line after it)."""
    a = cell.index("\n" + PKG_LINE) + 1
    b = cell.index("\n", a) + 1
    return cell[a:b], cell[:a] + cell[b + (cell[b:b + 1] == "\n"):]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str, hidden: bool = False) -> dict:
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}  # Colab; Jupyter
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def with_package(cell: str) -> tuple[dict, dict]:
    """The hidden cell "Package" (to run first) and the main cell without the package line."""
    from drybench import __version__

    line, rest = split_package(cell)
    package = code(f"#@title Package v{__version__} (run first; code hidden)\n"
                   "# The package, zipped and base64-encoded; the other cells extract it.\n" + line, hidden=True)
    return package, code(rest)


EXTRA_CELLS = [
    ("followup_checks.py", "## Details\nCPU only, after the main cell: tolerance check, filler by item design, "
                           "span entropy as a filler detector, completeness check, example spans. "
                           "At the end it zips the run folder and downloads it."),
    ("second_reader.py", "## Second reader\nScores the same responses with another reader model and compares "
                         "(~45 min on a T4)."),
    ("paper_tables_cell.py", "## Tables for the paper\nCPU only, after the cells above (or after restoring a "
                             "run): all tables for both readers in one Markdown file, plus a blind sample for "
                             "human labels; downloads them as a zip."),
]


def notebook(cell: str) -> dict:
    intro = ("# Filler in LLM answers: synthetic reports (T4)\n"
             "Runtime → Change runtime type → **T4 GPU**, then run *Package* (the code, hidden) and the main cell; "
             "Runtime → **Run all** also runs the cells after them. Settings are at the top of the main cell; "
             "rerunning resumes where it stopped.")
    package, main_cell = with_package(cell)
    restore =("## Optional: restore a saved run\n"
               "Only to continue a run saved earlier as a `.zip` by the Details cell (e.g. `t4_n20_v12.zip`): set "
               "`RESTORE = True`, run this cell and pick the file; it goes to Google Drive, and the main cell then "
               "reuses its responses. For a new run leave `RESTORE = False`: the cell does nothing.")
    restore_code = ("# Restore a run folder saved by the Details cell. RESTORE = False: nothing happens (a new run).\n"
                    "RESTORE = False  # True: upload a run's .zip saved earlier\n"
                    "TO_DRIVE = True  # False: into /content (lost when the runtime is recycled; set USE_DRIVE = False too)\n"
                    "import zipfile\n"
                    "from pathlib import Path\n\n"
                    "if not RESTORE:\n"
                    "    print(\"Nothing to restore (RESTORE = False): run the next cell.\")\n"
                    "else:\n"
                    "    from google.colab import files\n\n"
                    "    if TO_DRIVE:\n"
                    "        from google.colab import drive\n\n"
                    "        drive.mount(\"/content/drive\")\n"
                    "    dest = Path(\"/content/drive/MyDrive/drybench_runs\" if TO_DRIVE else \"/content/drybench_runs\")\n"
                    "    dest.mkdir(parents=True, exist_ok=True)\n"
                    "    for name in files.upload():\n"
                    "        zipfile.ZipFile(name).extractall(dest)\n"
                    "        print(\"restored\", name, \"→\", dest)\n")
    cells = [md(intro), package, md(restore), code(restore_code), main_cell]
    for name, text in EXTRA_CELLS:
        src = (ROOT / "colab" / name).read_text()
        if PKG_LINE in src and split_package(src)[0] == split_package(cell)[0]:
            src = split_package(src)[1]  # the same package: the cell "Package" serves this cell too
        cells += [md(text), code(src)]
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }


REAL_EXAMPLES = """# Examples: what the filler map deleted in real responses (after the main cell).
SHOW = 2        # responses per suite, most removable first
MODEL = next(iter(MODELS))
out = (Path("/content/drive/MyDrive/drybench_runs") if USE_DRIVE else Path(RUN_ROOT)) / RUN_NAME
pkg = install()
for suite in SUITES:
    print(f"\\n################ {suite}")
    try:
        run([sys.executable, pkg / "scripts" / "show_checks.py", "examples", out, "--source", suite,
             "--model", MODEL, "--variant", "default", "--examples", SHOW])
    except RuntimeError:
        pass  # nothing scored for this suite
"""
REAL_DOWNLOAD = """# Download the whole run (items, responses, scores, checks, tables) as one zip.
from google.colab import files

out = (Path("/content/drive/MyDrive/drybench_runs") if USE_DRIVE else Path(RUN_ROOT)) / RUN_NAME
files.download(f"{out}.zip")
"""


def real_notebook(cell: str) -> dict:
    intro = ("# Filler in LLM answers on real texts (T4)\n"
             "SQuAD 2.0 dev, MuSiQue dev, CUAD contract excerpts and Poddar et al.'s passage questions; "
             "small models that fit a T4. Runtime → Change runtime type → **T4 GPU**, then run *Package* (the code, "
             "hidden) and the main cell, or Runtime → **Run all**. Settings are at the top of the main cell; "
             "rerunning resumes where it stopped (with `USE_DRIVE = True` the run is kept in Google Drive).")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), *with_package(cell),
                  md("## Examples\nWhat the filler map deleted in a few responses per suite."),
                  code(REAL_EXAMPLES), md("## Download\nThe run folder as one zip."), code(REAL_DOWNLOAD)],
    }


def missing_notebook(cell: str) -> dict:
    intro = ("# Missing facts: which question to ask the reader (T4, ~30 min)\n"
             "Reads the real-text run in Google Drive (`RUN`) and compares ways of asking the reader about a fact "
             "the source does not state, for each reader in `READERS`. Runtime → Change runtime type → **T4 GPU**, "
             "then run *Package* (the code, hidden) and the main cell.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), *with_package(cell)],
    }


FINAL_DOWNLOAD = """# The samples for the human labels and the whole run as zip files.
out = (Path("/content/drive/MyDrive/drybench_runs") if USE_DRIVE else Path(RUN_ROOT).expanduser()) / RUN_NAME
try:
    from google.colab import files

    files.download(f"{out}_annotation.zip")
    # files.download(f"{out}.zip")  # the whole run (large)
except ImportError:  # a server: the zips are on its disk
    print("For the annotators:", f"{out}_annotation.zip")
    print("Whole run:", f"{out}.zip")
"""


def final_notebook(cell: str) -> dict:
    intro = ("# Filler in LLM answers: the confirmation run (A100)\n"
             "One frozen version of the code (see `docs/PROTOCOL.md` in the package), new items, three models, "
             "three readers, every check, the LLM judge and Selective Context, and the samples for human labels. "
             "Set `RUN_ROOT` and `LOCAL` (a server with local model folders) at the top of the main cell, then run "
             "*Package* (the code, hidden) and the main cell; rerunning resumes. On Colab: an A100 runtime, "
             "`USE_DRIVE = True`, `PIP = True`, `HF_TOKEN` in the Colab secrets.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), *with_package(cell), md("## Download\nThe annotation folder as one zip."),
                  code(FINAL_DOWNLOAD)],
    }


DEV_DOWNLOAD = """# Download both development runs with the baselines, the tables and the confirmatory statistics.
root = Path("/content/drive/MyDrive/drybench_runs") if USE_DRIVE else Path(RUN_ROOT).expanduser()
try:
    from google.colab import files

    files.download(str(root / f"{ZIP_NAME}.zip"))
except ImportError:  # a server: the zip is on its disk
    print("The zip:", root / f"{ZIP_NAME}.zip")
"""


def dev_baselines_notebook(cell: str) -> dict:
    intro = ("# Filler in LLM answers: the development runs on the confirmation run's machine (A100)\n"
             "After the T4 development runs of this package version (synthetic reports and real texts): set "
             "`RUN_ROOT` (the folder with both runs) and `LOCAL` (local model folders) at the top of the main cell, "
             "run *Package* (the code, hidden) and the main cell. It describes the local models, scores a sample "
             "of both runs' responses again with the main reader and compares, adds the judge (Qwen2.5-14B) and "
             "Selective Context (Qwen2.5-3B) to both runs, remakes their tables and writes the development values "
             "of the confirmatory tests (`docs/PROTOCOL.md`). On Colab: an A100 runtime, `USE_DRIVE = True`, "
             "`PIP = True`.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), *with_package(cell), md("## Download\nBoth runs and the statistics as one zip."),
                  code(DEV_DOWNLOAD)],
    }


def probe_notebook(cell: str) -> dict:
    intro = ("# Filler in LLM answers: which local models the benchmark can use (a server)\n"
             "Set `MODELS` (local model folders) at the top of the main cell, then run *Package* (the code, hidden) "
             "and the main cell. Each model: does it load, does it answer in plain text, can it serve as reader and "
             "as judge, how much GPU memory it takes. Nothing is downloaded or installed.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), *with_package(cell)],
    }


def main() -> None:
    out = ROOT / "colab"
    cell = build_cell()
    (out / "drybench_colab_cell.py").write_text(cell)
    (out / "paper_tables_cell.py").write_text(build_cell(template="paper_tables_template.py"))
    (out / "drybench_colab.ipynb").write_text(json.dumps(notebook(cell), indent=1))
    real = build_cell(template="real_template.py")
    (out / "real_texts_cell.py").write_text(real)
    (out / "real_texts_colab.ipynb").write_text(json.dumps(real_notebook(real), indent=1))
    miss = build_cell(template="missing_check_template.py")
    (out / "missing_check_colab.ipynb").write_text(json.dumps(missing_notebook(miss), indent=1))
    final = build_cell(template="final_template.py")
    (out / "final_a100_cell.py").write_text(final)
    (out / "final_a100.ipynb").write_text(json.dumps(final_notebook(final), indent=1))
    dev = build_cell(template="dev_baselines_template.py")
    (out / "dev_baselines_a100.ipynb").write_text(json.dumps(dev_baselines_notebook(dev), indent=1))
    probe = build_cell(template="probe_template.py")
    (out / "probe_models.ipynb").write_text(json.dumps(probe_notebook(probe), indent=1))
    print(f"cell: {len(cell) / 1024:.0f} KB, real-text cell: {len(real) / 1024:.0f} KB → {out}")


if __name__ == "__main__":
    main()
