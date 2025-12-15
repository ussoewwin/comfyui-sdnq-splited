#!/usr/bin/env python3
"""
Count characters in a text file or stdin.

Counts are based on Python Unicode code points (len()).
Outputs:
- total_chars: includes all characters (including newlines)
- chars_no_newlines: excludes '\r' and '\n'
- chars_no_whitespace: excludes all Unicode whitespace (str.isspace())
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def read_text_from_stdin() -> str:
    return sys.stdin.read()


def read_text_from_file(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="strict")


def main() -> int:
    p = argparse.ArgumentParser(description="Count characters in a file or stdin.")
    p.add_argument("path", nargs="?", help="Text file path. If omitted, read from stdin.")
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8). Ignored when reading from stdin.",
    )
    args = p.parse_args()

    if args.path:
        path = Path(args.path)
        text = read_text_from_file(path, encoding=args.encoding)
    else:
        text = read_text_from_stdin()

    total_chars = len(text)
    chars_no_newlines = len(text.replace("\r", "").replace("\n", ""))
    chars_no_whitespace = sum(1 for ch in text if not ch.isspace())

    print(f"total_chars={total_chars}")
    print(f"chars_no_newlines={chars_no_newlines}")
    print(f"chars_no_whitespace={chars_no_whitespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


