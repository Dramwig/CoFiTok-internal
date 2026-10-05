# CoFiTok LaTeX Draft

This directory contains the venue-neutral CoFiTok LaTeX working version.

Figure 2 uses the same four standalone PNG subfigures as the AAAI-27 build and
assembles them in LaTeX; do not replace them with a precomposed panel image.

Current source and shared files:

```text
main.tex
../references.bib
../citation_audit.md
```

Versioned tables and figures are read directly from
`../../CoFiTok-internal/artifacts/`.

Build locally with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

or:

```bash
make
```

This is not the submission template. The canonical AAAI-27 entrypoints are in
`../venues/aaai27/`.
