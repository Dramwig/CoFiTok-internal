# CoFiTok ICLR 2027 submission build

This directory contains the anonymous ICLR 2027 paper source. The technical
appendix is part of the same submission PDF and is inserted after the
references. The paper uses the official `iclr2027_conference` style package and
keeps the locked CoFiTok evidence package as the numerical source of tables.

## Build

Run the commands from this directory so that the relative figure, table, and
evidence paths resolve correctly. All generated files are written below
`build/`; the source directory stays free of auxiliary files and PDFs:

```powershell
$buildDir = "build"
New-Item -ItemType Directory -Force $buildDir | Out-Null

pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory $buildDir main_iclr2027.tex

$savedBibInputs = $env:BIBINPUTS
$savedBstInputs = $env:BSTINPUTS
try {
  $sourceDir = (Get-Location).Path
  $env:BIBINPUTS = $sourceDir + ";"
  $env:BSTINPUTS = $sourceDir + ";"
  Push-Location $buildDir
  bibtex main_iclr2027
} finally {
  Pop-Location
  $env:BIBINPUTS = $savedBibInputs
  $env:BSTINPUTS = $savedBstInputs
}

pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory $buildDir main_iclr2027.tex
pdflatex -interaction=nonstopmode -halt-on-error `
  -output-directory $buildDir main_iclr2027.tex

```

The final PDF is `build/main_iclr2027.pdf`. It contains the main text,
references, and the technical appendix in that order.

The main text must remain within the ICLR 2027 nine-page main-text limit;
references and the appendix follow the main text in the same PDF.

The `build/` directory contains local auxiliary files and PDFs. A clean build
should be performed from a fresh copy before submission, with the source tree
kept separate from the locked experiment artifacts.
