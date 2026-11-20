# Paper (IEEE conference format)

`main.tex` and `references.bib` are the LaTeX version of [docs/PAPER_DRAFT.md](../docs/PAPER_DRAFT.md); every number traces to a section of [docs/RESULTS.md](../docs/RESULTS.md).

- Figures come from `python -m src.evaluation.make_figures` (written to `docs/figures/`). `main.tex` looks in `figures/` first, then `../docs/figures/`, so it builds inside the repository as is. For Overleaf or a standalone copy, put `fig_graph_effect.pdf` and `fig_topical_ndcg.pdf` in a `figures/` folder next to `main.tex`.
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or `tectonic main.tex`). The architecture figure is TikZ.
- Not yet compiled on the machine that made the November 20 update (no TeX installed); check the PDF for overfull tables before submission.
- Still placeholders: authors and affiliations. Two references are unconfirmed (see the comment at the top of `references.bib`).
