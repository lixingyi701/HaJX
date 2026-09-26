#!/bin/bash
# 单章编译：只编译某一问的正文章节，复用 main.tex 导言区。
# 用法：./compile_chapter.sh q1|q2|q3|q4|<sec 文件名.tex>
# 产物：pdf/<章名>/<章名>_<MMDD>-<HHMM>.pdf
# 单章附参考文献以解析 \cite；跨章 \ref 需在正文中避免或整本编译核对。
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/bin:$PATH"

case "${1:-}" in
  q1) CH=q1; SRC=sec06_q1.tex;;
  q2) CH=q2; SRC=sec07_q2.tex;;
  q3) CH=q3; SRC=sec08_q3.tex;;
  q4) CH=q4; SRC=sec09_q4.tex;;
  *.tex) CH="${1%.tex}"; SRC="$1";;
  *) echo "用法: $0 q1|q2|q3|q4|<sec文件.tex>" >&2; exit 1;;
esac
[[ -f "$SRC" ]] || { echo "找不到 $SRC" >&2; exit 1; }

WRAP="chapter_${CH}.tex"
sed '/^\\begin{document}/q' main.tex > "$WRAP"
printf '\\input{%s}\n\\input{references.tex}\n\\end{document}\n' "$SRC" >> "$WRAP"

TEXINPUTS=.: xelatex -interaction=nonstopmode "$WRAP" > /dev/null
TEXINPUTS=.: xelatex -interaction=nonstopmode "$WRAP" 2>&1 | tail -5
if grep -Eq 'LaTeX Warning: (Citation|Reference).*undefined|There were undefined (references|citations)' "chapter_${CH}.log"; then
  echo "编译完成但仍有未解析的引用，见 chapter_${CH}.log" >&2
  exit 1
fi

OUT_DIR="pdf/$CH"
mkdir -p "$OUT_DIR"
TS=$(date +%m%d-%H%M)
OUTPUT="$OUT_DIR/${CH}_${TS}.pdf"
[[ -e "$OUTPUT" ]] && OUTPUT="$OUT_DIR/${CH}_$(date +%m%d-%H%M-%S).pdf"
mv "chapter_${CH}.pdf" "$OUTPUT"
rm -f "chapter_${CH}.aux" "chapter_${CH}.out" "chapter_${CH}.log" "$WRAP"
echo "Compiled: $OUTPUT"
