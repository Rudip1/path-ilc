#!/usr/bin/env bash
# Build the reports from source (here) straight into ../technical_report/ with
# their final names. Run from anywhere; outputs:
#   ../technical_report/full_technical_report.pdf      (from technical_report.tex)
#   ../technical_report/summary_technical_report.pdf   (from summary_a4.tex)
# Intermediate .aux/.log/.out are written to the output dir and removed at the end.
set -e
cd "$(dirname "$0")"
OUT=../technical_report
P="pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$OUT"

# full report (twice: resolve refs/citations); summary (twice for layout)
$P -jobname=full_technical_report    technical_report.tex >/dev/null
$P -jobname=full_technical_report    technical_report.tex >/dev/null
$P -jobname=summary_technical_report summary_a4.tex        >/dev/null
$P -jobname=summary_technical_report summary_a4.tex        >/dev/null

rm -f "$OUT"/*.aux "$OUT"/*.log "$OUT"/*.out
echo "Built: $OUT/full_technical_report.pdf, $OUT/summary_technical_report.pdf"
