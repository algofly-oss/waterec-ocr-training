"""Validate publishable source and notebooks, excluding local experiment files."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for folder in ("waterec_ocr", "notebooks", "tools", "configs", "tests", "docs", "results", "annotator"):
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix not in (
                ".py", ".md", ".yaml", ".json", ".ipynb", ".txt", ".html"
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"gh[pousr]_\w{20,}|github_pat_\w{30,}|af_[a-z0-9]{15,}", text):
                raise ValueError(f"Credential pattern in {path.relative_to(ROOT)}")
            for prefix in ("/" + x + "/" for x in ("app", "mnt", "root")):
                if prefix in text:
                    raise ValueError(f"Local experiment path in {path.relative_to(ROOT)}")
            if path.suffix == ".py":
                ast.parse(text, filename=str(path))
            if path.suffix == ".ipynb":
                notebook = json.loads(text)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        if cell.get("outputs") or cell.get("execution_count") is not None:
                            raise ValueError(f"Saved output in {path.name}")
                        ast.parse("".join(cell["source"]), filename=path.name)
            count += 1
    print(f"Checked {count} public source/documentation files")


if __name__ == "__main__":
    main()
