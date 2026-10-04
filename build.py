"""Rebuild and check every deliverable.

    python build.py              tests, correction, notebook, monitor, slides, then the checks
    python build.py --no-slides  skip the Marp render (it needs Node and a Chromium browser)

Order matters: model_corrige.py and the notebook produce the figures the slides embed.
"""
import argparse
import shutil
import subprocess
import sys

import jupytext
import nbformat
from nbclient import NotebookClient

from equialgo.data import ROOT
from equialgo.validate import check_file

MARP = "@marp-team/marp-cli@4"
REQUIRED = ["predictions.csv", "audit_rapport.ipynb", "model_corrige.py", "presentation.pdf"]
SUPPORTING = ["gouvernance.md", "monitoring.py", "figures/pareto.png", "figures/pareto_resume.png",
              "figures/monitoring.png", "figures/decomposition.png", "figures/sans_region.png",
              "figures/methode.png", "figures/recherche.png"]


def step(title):
    print(f"\n=== {title} ===", flush=True)


def run(command, **kwargs):
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def build_notebook(source="audit_rapport.py", output="audit_rapport.ipynb"):
    """Convert the cell-format script to a notebook, execute it, save it with its outputs."""
    notebook = jupytext.read(ROOT / source)
    NotebookClient(notebook, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    # Keep the saved notebook stable between builds: no timings, no random cell ids.
    notebook.metadata.pop("jupytext", None)
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{index:03d}"
        cell.metadata.pop("execution", None)
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / output)


def build_slides(source="presentation.md", output="presentation.pdf"):
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx not found: install Node to render the slides")
    # --no-stdin and a closed stdin: otherwise marp waits for piped input and hangs.
    run([npx, "--yes", MARP, "--no-stdin", source, "--pdf", "--allow-local-files", "-o", output],
        stdin=subprocess.DEVNULL, timeout=300)


def check_deliverables(slides=True):
    """Every deliverable exists and is valid. Returns the list of problems."""
    problems = []
    for name in REQUIRED + SUPPORTING:
        path = ROOT / name
        if name == "presentation.pdf" and not slides:
            continue
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"{name}: missing or empty")
    try:
        rate = check_file()
        print(f"predictions.csv: valid, grant rate {rate:.2%}")
    except Exception as error:  # report every problem, not only the first
        problems.append(f"predictions.csv: {error}")
    notebook_path = ROOT / "audit_rapport.ipynb"
    if notebook_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
        code = [c for c in notebook.cells if c.cell_type == "code"]
        if not code or any(not c.outputs for c in code):
            problems.append("audit_rapport.ipynb: a code cell has no output")
        if any(o.output_type == "error" for c in code for o in c.outputs):
            problems.append("audit_rapport.ipynb: a cell raised an error")
    pdf_path = ROOT / "presentation.pdf"
    if slides and pdf_path.exists() and pdf_path.read_bytes()[:5] != b"%PDF-":
        problems.append("presentation.pdf: not a PDF")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-slides", action="store_true")
    args = parser.parse_args()

    step("Tests")
    run([sys.executable, "-m", "pytest", "-q", "tests"])
    step("model_corrige.py -> figures/pareto.png, predictions.csv")
    run([sys.executable, "model_corrige.py"])
    step("audit_rapport.py -> audit_rapport.ipynb")
    build_notebook()
    step("monitoring.py -> figures/monitoring.png")
    run([sys.executable, "monitoring.py"])
    if not args.no_slides:
        step("presentation.md -> presentation.pdf")
        build_slides()

    step("Checks")
    problems = check_deliverables(slides=not args.no_slides)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        sys.exit(1)
    print("All deliverables built and valid.")


if __name__ == "__main__":
    main()
