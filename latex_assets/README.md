# Report sources (`latex_assets/`)

LaTeX sources and TikZ figures for the two write-ups. The compiled PDFs are
written to **`../technical_report/`** (the deliverables live there, not here):

| Source (here) | Builds | Use it as |
|---|---|---|
| `technical_report.tex` | `../technical_report/full_technical_report.pdf` — 4-page two-column report: abstract, intro, model (with equations), method (with the network-architecture figure), results (table + figures), full-width analysis gallery, honest discussion, DOI-linked references. | writing sample / deeper read |
| `summary_a4.tex` | `../technical_report/summary_technical_report.pdf` — 2-page A4 landscape. **P1**: background, the flexible-joint model, method, 3 headline results (each with a "why it matters" line). **P2**: figure appendix. | quick teaser / attachment |

### Building blocks (shared TikZ figures, `\includegraphics`-d by both)
| File | What |
|---|---|
| `model_schematic.pdf` / `.tex` | flexible-joint drivetrain schematic (real KUKA render + TikZ, B&W) |
| `method_loop.pdf` / `.tex` | path-ILC + learned-layer control-loop block diagram (TikZ) |
| `network_architecture.pdf` / `.tex` | the learned correction layer (MLP) diagram (TikZ) |
| `kuka_arm.png` / `render_arm.py` | grayscale iiwa14 render used in the schematic, and its regen script |

## Rebuild (needs LaTeX + the result PNGs in `../outputs/`)

```bash
cd latex_assets
./build.sh        # builds both PDFs straight into ../technical_report/ with final names
```

`build.sh` runs `pdflatex` with `-output-directory=../technical_report` and
`-jobname` set to the final names, runs each twice (refs/layout), and cleans up
the `.aux/.log/.out`. If you change a TikZ figure, recompile it first
(`pdflatex model_schematic.tex`, etc.) so its `.pdf` updates.

Numbers come from `python src/run_all.py` (seeded/reproducible).
