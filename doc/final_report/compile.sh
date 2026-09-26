#!/bin/bash
# 编译 final_report/main.tex 并把输出 PDF 重命名为 main_<MMDD>-<HHMM>.pdf
# 用法：./compile.sh
set -euo pipefail
cd "$(dirname "$0")"
TEXINPUTS=.: xelatex -interaction=nonstopmode main.tex 2>&1 | tail -10
TEXINPUTS=.: xelatex -interaction=nonstopmode main.tex 2>&1 | tail -10
TS=$(date +%m%d-%H%M)
OUT_DIR="pdf"
mkdir -p "$OUT_DIR"
OUTPUT="$OUT_DIR/main_${TS}.pdf"
if [[ -e "$OUTPUT" ]]; then
  TS=$(date +%m%d-%H%M-%S)
  OUTPUT="$OUT_DIR/main_${TS}.pdf"
fi
mv main.pdf "$OUTPUT"
echo "Compiled: $OUTPUT"
