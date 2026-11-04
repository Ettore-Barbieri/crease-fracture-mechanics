#!/bin/sh
# Rebuild the figure PDFs in figures/standalone/ from the pgfplots/TikZ sources and figures/data/.
# Usage: ./build_figures.sh             (all figures)
#        ./build_figures.sh pgf_fs_Gf   (one figure)
cd "$(dirname "$0")/figures/standalone" || exit 1
if [ $# -gt 0 ]; then list="$@"; else list=$(ls *.tex | sed 's/\.tex$//'); fi
for f in $list; do
  echo "building $f"
  pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" >/dev/null || echo "  FAILED: see figures/standalone/$f.log"
done
rm -f *.aux *.log
