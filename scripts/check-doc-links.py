"""Verify local Markdown link targets in checked-in documentation."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
docs = [*root.glob("*.md"), *root.glob("docs/**/*.md"), *root.glob("apps/**/*.md")]
docs.extend(root.glob("deployment/**/*.md"))
missing = []
for path in docs:
    for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
        target = target.split("#", 1)[0]
        if target and "://" not in target and not (path.parent / target).exists():
            missing.append(f"{path.relative_to(root)} -> {target}")
if missing:
    raise SystemExit("\n".join(missing))
print(f"Checked local links in {len(docs)} Markdown files")
