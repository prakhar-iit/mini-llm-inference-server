"""Build docs/course/index.html from docs/course/src/.

The course is one HTML page: 01-shell.html (title, styles, layout), then every src/*.js in name order
inside one <script>. Content lives in the .js files as data (UNITS, ITEMS via M() and D(), ORIENTATION,
APPENDIX); 06-app.js renders it.

    python docs/course/build.py          # writes docs/course/index.html
    python docs/course/build.py --check  # fails if index.html is out of date with src/

After building, republish the artifact (see CLAUDE.md, "Course").
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE / "index.html"


def build() -> str:
    shell = (SRC / "01-shell.html").read_text()
    scripts = "".join(p.read_text() for p in sorted(SRC.glob("*.js")))
    return f"{shell}<script>\n{scripts}</script>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if index.html differs from a fresh build")
    args = ap.parse_args()
    html = build()
    if args.check:
        if not OUT.exists() or OUT.read_text() != html:
            sys.exit("docs/course/index.html is out of date: run python docs/course/build.py")
        return
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(HERE.parent.parent)} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
