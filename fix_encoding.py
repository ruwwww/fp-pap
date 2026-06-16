"""fix_encoding.py - one-time fix to add encoding='utf-8' to all open() calls"""
import re
from pathlib import Path

files = [
    "src/artifacts/artifact_store.py",
    "src/artifacts/run_record.py",
    "scripts/train.py",
    "scripts/search.py",
    "scripts/forecast.py",
    "scripts/runs.py",
    "scripts/evaluate.py",
]

def fix_open(text):
    """Add encoding='utf-8' to open() calls that don't already have it."""
    result = []
    for line in text.splitlines():
        # Skip lines that already have encoding=
        if "encoding=" in line:
            result.append(line)
            continue
        # Match: open(something) as f: or open(something, 'r') etc.
        # Add encoding='utf-8' to read opens
        # Don't touch: open(..., 'wb'), open(..., 'rb'), open(..., 'w') — write with no encoding issue
        if "open(" in line and "as f:" in line and "\"w\"" not in line and "'w'" not in line and "wb" not in line and "rb" not in line:
            line = line.replace(") as f:", ", encoding=\"utf-8\") as f:")
        result.append(line)
    return "\n".join(result)

for fpath in files:
    p = Path(fpath)
    if not p.exists():
        print(f"SKIP: {fpath}")
        continue
    text = p.read_text(encoding="utf-8")
    new_text = fix_open(text)
    if new_text != text:
        p.write_text(new_text, encoding="utf-8")
        print(f"Fixed: {fpath}")
    else:
        print(f"OK   : {fpath}")
