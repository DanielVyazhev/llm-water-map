"""Build the self-contained Colab cell and notebook from colab/cell_template.py.

The drybench package (code + pyproject + README) is zipped, base64-encoded and embedded in
the cell, so the notebook needs no uploads.

  python scripts/make_colab.py            → colab/drybench_colab.ipynb, colab/drybench_colab_cell.py,
                                            real_texts_colab.ipynb, missing_check_colab.ipynb, final_a100.ipynb
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["pyproject.toml", "README.md", "scripts/pilot_validation.py", "scripts/run_real.py",
                  "scripts/run_synthetic.py", "scripts/run_final.py", "scripts/show_checks.py",
                  "scripts/trace_response.py", "scripts/chat_annotators.py", "scripts/check_abstain.py",
                  "annotation/dry_span_labels_offline.html", "docs/ANNOTATION.md", "docs/PROTOCOL.md",
                  "results/abstain_labels.json"]:
            z.write(ROOT / f, f)
        for p in sorted((ROOT / "drybench").rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
    return buf.getvalue()


def build_cell(overrides: str | None = None, template: str = "cell_template.py") -> str:
    from drybench import __version__

    tmpl = (ROOT / "colab" / template).read_text()
    cell = tmpl.replace("__VERSION__", __version__).replace(
        "__PKG_B64__", base64.b64encode(package_zip()).decode())
    if overrides is not None:  # replace the settings block (used for local dry runs)
        start, end = "# ---- settings ----\n", "# ---- end settings ----\n"
        a, b = cell.index(start) + len(start), cell.index(end)
        cell = cell[:a] + overrides + cell[b:]
    return cell


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
             "Runtime → Change runtime type → **T4 GPU**, then run the cell below. "
             "Settings are at the top of the cell; rerunning resumes where it stopped.")

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    restore = ("## Optional: restore a saved run\n"
               "In a new Colab session, run this cell and pick the `.zip` that the Details cell downloaded "
               "(e.g. `t4_n20.zip`); it goes to Google Drive, and the main cell then reuses its responses "
               "instead of generating them again.")
    restore_code = ("# Restore a run folder saved by the Details cell (skip in the session that made it).\n"
                    "TO_DRIVE = True  # False: into /content (lost when the runtime is recycled; set USE_DRIVE = False too)\n"
                    "import zipfile\n"
                    "from pathlib import Path\n"
                    "from google.colab import files\n\n"
                    "if TO_DRIVE:\n"
                    "    from google.colab import drive\n\n"
                    "    drive.mount(\"/content/drive\")\n"
                    "dest = Path(\"/content/drive/MyDrive/drybench_runs\" if TO_DRIVE else \"/content/drybench_runs\")\n"
                    "dest.mkdir(parents=True, exist_ok=True)\n"
                    "for name in files.upload():\n"
                    "    zipfile.ZipFile(name).extractall(dest)\n"
                    "    print(\"restored\", name, \"→\", dest)\n")
    cells = [md(intro), md(restore), code(restore_code), code(cell)]
    for name, text in EXTRA_CELLS:
        cells += [md(text), code((ROOT / "colab" / name).read_text())]
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
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    intro = ("# Filler in LLM answers on real texts (T4)\n"
             "SQuAD 2.0 dev, MuSiQue dev, CUAD contract excerpts and Poddar et al.'s passage questions; "
             "small models that fit a T4. Runtime → Change runtime type → **T4 GPU**, then run the cell below. "
             "Settings are at the top of the cell; rerunning resumes where it stopped (with `USE_DRIVE = True` "
             "the run is kept in Google Drive).")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), code(cell), md("## Examples\nWhat the filler map deleted in a few responses per suite."),
                  code(REAL_EXAMPLES), md("## Download\nThe run folder as one zip."), code(REAL_DOWNLOAD)],
    }


def missing_notebook(cell: str) -> dict:
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    intro = ("# Missing facts: which question to ask the reader (T4, ~30 min)\n"
             "Reads the real-text run in Google Drive (`RUN`) and compares ways of asking the reader about a fact "
             "the source does not state, for each reader in `READERS`. Runtime → Change runtime type → **T4 GPU**, "
             "then run the cell.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), code(cell)],
    }


FINAL_DOWNLOAD = """# Download the samples for the human labels (and, if needed, the whole run) as zip files.
from google.colab import files

out = (Path("/content/drive/MyDrive/drybench_runs") if USE_DRIVE else Path(RUN_ROOT)) / RUN_NAME
files.download(f"{out}_annotation.zip")
# files.download(f"{out}.zip")  # the whole run (large)
"""


def final_notebook(cell: str) -> dict:
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    intro = ("# Filler in LLM answers: the confirmation run (A100)\n"
             "One frozen version of the code (see `docs/PROTOCOL.md` in the package), new items, seven models, "
             "three readers, every check, the LLM judge and Selective Context, and the samples for human labels. "
             "Runtime → Change runtime type → **A100 GPU**; put `HF_TOKEN` into the Colab secrets; run the cell. "
             "Rerunning resumes from Google Drive.")
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [md(intro), code(cell), md("## Download\nThe annotation folder as one zip."), code(FINAL_DOWNLOAD)],
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
    print(f"cell: {len(cell) / 1024:.0f} KB, real-text cell: {len(real) / 1024:.0f} KB → {out}")


if __name__ == "__main__":
    main()
