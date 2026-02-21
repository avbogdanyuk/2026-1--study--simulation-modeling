.PHONY: all clean cleanall pdf docx

all: pdf docx

pdf:
	@quarto render report.qmd --to pdf

docx:
	@quarto render report.qmd --to docx

clean:
	-rm -rf _output

cleanall: clean
	-rm -rf .quarto