# Makefile for COGS 107 Lectures

# Compiler
PDFLATEX = pdflatex
BIBTEX = bibtex

# Flags for pdflatex
# -interaction=nonstopmode: Don't stop on errors, just report them
# -file-line-error: Show errors in C-style file:line:error format
# -output-directory=build: Put auxiliary files in build/
# -jobname=<name>: Set the base name for output files
LFLAGS = -interaction=nonstopmode -file-line-error

# Directories
TEX_DIR = tex
BUILD_DIR = .build
PDF_DIR = pdf

# Input files for BibTeX
# Search current dir and tex dir (and subdirs) for .bib files
BIBINPUTS = ./:$(TEX_DIR)//:

# Find all .tex files in the source directory
SOURCES = $(wildcard $(TEX_DIR)/*.tex)

# Generate PDF target names from source names
# Replace TEX_DIR with PDF_DIR and .tex with .pdf
PDFS = $(patsubst $(TEX_DIR)/%.tex,$(PDF_DIR)/%.pdf,$(SOURCES))

# Define TeX source files (assuming you have others)
TEX_FILES := $(wildcard tex/*.tex)
# Convert TeX source files to PDF targets
PDF_FILES := $(patsubst tex/%.tex,pdf/%.pdf,$(TEX_FILES))

# Phony targets (targets that don't represent files)
.PHONY: all clean watch

# Default target: Build all PDFs and necessary figures
# The figure dependency is handled by the specific rule below
all: $(PDF_FILES)

# General rule to build a PDF from a .tex file
$(PDF_DIR)/%.pdf: $(TEX_DIR)/%.tex $(TEX_DIR)/beamertheme-custom.sty $(TEX_DIR)/symbols-custom.sty
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(PDF_DIR)
	@echo "Building $@ from $< ..."
	@# 1. Initial pdflatex run (generates .aux file)
	TEXINPUTS=./$(TEX_DIR)//: $(PDFLATEX) $(LFLAGS) -output-directory=$(BUILD_DIR) -jobname=$(basename $(notdir $<)) $<
	@# 2. Check if bibtex is needed and run it if so
	@if grep -q '\\bibdata' $(BUILD_DIR)/$(basename $(notdir $<)).aux; then \
		echo "--> Found bibliography, running BibTeX on $(basename $(notdir $<)).aux..."; \
		BIBINPUTS=$(BIBINPUTS) $(BIBTEX) $(BUILD_DIR)/$(basename $(notdir $<)) > $(BUILD_DIR)/bibtex.log ; \
	else \
		echo "--> No bibliography found in $(basename $(notdir $<)).aux, skipping BibTeX."; \
	fi
	@# 3. Second pdflatex run (to incorporate bibliography .bbl file if created)
	TEXINPUTS=./$(TEX_DIR)//: $(PDFLATEX) $(LFLAGS) -output-directory=$(BUILD_DIR) -jobname=$(basename $(notdir $<)) $<
	@# 4. Third pdflatex run (to resolve citations if needed)
	TEXINPUTS=./$(TEX_DIR)//: $(PDFLATEX) $(LFLAGS) -output-directory=$(BUILD_DIR) -jobname=$(basename $(notdir $<)) $<
	@# 5. Move the final PDF
	@mv $(BUILD_DIR)/$(basename $(notdir $<)).pdf $@
	@echo "Finished building $@"

# Clean up build files
clean:
	@echo "Cleaning up build files..."
	rm -rf $(BUILD_DIR)

# Add .PRECIOUS rule to keep intermediate files in .build if needed for debugging
# .PRECIOUS: $(BUILD_DIR)/%.aux $(BUILD_DIR)/%.bbl $(BUILD_DIR)/%.blg

# Watch for changes and recompile
watch:
	while inotifywait -e modify -r tex/* tex/figures/* scripts/*; do \
		make all; \
	done


# Rule to create the figures directory (order-only prerequisite)
tex/figures:
	mkdir -p $@
