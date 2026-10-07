# pdflatex needs -shell-escape for minted (pygments must be installed: pip install pygments)
LATEX = pdflatex -shell-escape -interaction=nonstopmode -halt-on-error
all: ms.pdf
ms.pdf: ms.tex macros.tex refs.bib sections/*.tex
	$(LATEX) ms.tex && bibtex ms && $(LATEX) ms.tex && $(LATEX) ms.tex
quick:
	$(LATEX) ms.tex
clean:
	rm -rf ms.aux ms.bbl ms.blg ms.log ms.out ms.pdf _minted-ms ms.pyg
.PHONY: all quick clean
