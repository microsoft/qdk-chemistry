# Makefile for building all documentation

################################################################################
# Top-level configuration
################################################################################

# Docs source directory
SOURCE_DIR = source
# Docs output directory
BUILD_DIR = build
# Directory for Doxygen-generated XML files
DOXYGEN_API_DIR = source/api/doxygen
# Directory for Sphinx-generated API documentation
SPHINX_API_DIR = source/api/api_autogen
# Directory for Breathe-generated API documentation
BREATHE_API_DIR = source/api/breathe_api_autogen
# Python source directory (for Sphinx API generation)
PYTHON_SRC_DIR = ../python/src
# Format of generated documentation (can override from environment)
SPHINX_FORMAT ?= html

.PHONY: check help clean doxygen breathe sphinx-api sphinx-autosummary sphinx-docs clean-doxygen clean-breathe clean-sphinx-api clean-sphinx-autosummary clean-sphinx-docs

default: check
	@echo "Try running 'make clean' to clean things up or 'make all' to build everything."

check:
	@if [ "$(shell pwd)" = "$(abspath $(dir $(lastword $(MAKEFILE_LIST))))" ]; then \
		echo "Running in correct directory: $(abspath $(dir $(lastword $(MAKEFILE_LIST))))"; \
	else \
		echo "Error: Please run 'make' from the docs directory: \
		$(abspath $(dir $(lastword $(MAKEFILE_LIST))))" && exit 1; \
	fi

clean: check clean-doxygen clean-breathe clean-sphinx-api clean-sphinx-autosummary clean-sphinx-docs
	@echo "** Done cleaning."

all: check sphinx-docs
	@echo "** Built all documentation."

################################################################################
# Doxygen XML build
################################################################################

DOXYGEN_DEBUG = doxygen-output.txt
# Extra options for Doxygen command (can be specified in environment)
DOXYGEN_OPTIONS ?=

doxygen: Doxyfile
	@echo "** Generating Doxygen XML documentation..."
	doxygen $(DOXYGEN_OPTIONS) Doxyfile > $(DOXYGEN_DEBUG) 2>&1
	@echo "** Doxygen XML documentation generated."
	@echo "** Saved output to $(DOXYGEN_DEBUG)."
	@count=$$(grep -E -i "warning|error" $(DOXYGEN_DEBUG) | wc -l); \
	echo "** Found $$count warnings/errors in the Doxygen output."
	@echo


clean-doxygen: check
	@echo "** Cleaning Doxygen-generated files..."
	rm -rf $(DOXYGEN_API_DIR)
	rm -f $(DOXYGEN_DEBUG)
	@echo

################################################################################
# Breathe API build
################################################################################

BREATHE_DEBUG = breathe-output.txt
BREATHE_OPTIONS = -f -m -g namespace -p "QDK/Chemistry"
# Extra options for Breathe command (can be specified in environment)
BREATHE_EXTRA_OPTIONS ?=

breathe: doxygen
	@echo "** Generating Breathe API documentation..."
	breathe-apidoc $(BREATHE_OPTIONS) $(BREATHE_EXTRA_OPTIONS) \
		-o $(BREATHE_API_DIR) $(DOXYGEN_API_DIR)/xml > $(BREATHE_DEBUG) 2>&1
	@echo "** Breathe API documentation generated."
	@echo "** Saved output to $(BREATHE_DEBUG)."
	@echo

clean-breathe: check
	@echo "** Cleaning Breathe-generated files..."
	rm -rf $(BREATHE_API_DIR)
	rm -f $(BREATHE_DEBUG)
	@echo

################################################################################
# Sphinx API build
################################################################################

# The sphinx-apidoc command doesn't generate any output; including for consistency
SPHINX_API_DEBUG = sphinx-api-output.txt
SPHINX_API_OPTIONS = -f --separate --module-first --private --implicit-namespaces --doc-project='QDK/Chemistry Python API'
# Extra options for Sphinx command (can be specified in environment)
SPHINX_API_EXTRA_OPTIONS ?=

sphinx-api:
	@echo "** Building Sphinx API documentation..."
	sphinx-apidoc $(SPHINX_API_OPTIONS) $(SPHINX_API_EXTRA_OPTIONS) \
		-o $(SPHINX_API_DIR) $(PYTHON_SRC_DIR)/qdk_chemistry | tee $(SPHINX_API_DEBUG) 2>&1
	@echo "** Sphinx API documentation generated."
	@echo "** Saved output to $(SPHINX_API_DEBUG)."
	@count=$$(grep -E "WARNING|ERROR" $(SPHINX_API_DEBUG) | wc -l); \
	echo "** Found $$count warnings/errors in the Sphinx API output."
	@echo

clean-sphinx-api: check
	@echo "** Cleaning Sphinx API-generated files..."
	rm -rf $(SPHINX_API_DIR)
	rm -f $(SPHINX_API_DEBUG)
	@echo

################################################################################
# Sphinx autosummary build
################################################################################

SPHINX_SUMMARY_DEBUG = sphinx-autosummary-output.txt
SPHINX_SUMMARY_WARNINGS = sphinx-autosummary-warnings.txt
# Using more than one thread can cause issues interpreting the output
SPHINX_SUMMARY_OPTIONS = -j 1 -n -w $(SPHINX_SUMMARY_WARNINGS)
# Extra options for Sphinx command (can be specified in environment)
SPHINX_SUMMARY_EXTRA_OPTIONS ?=

sphinx-autosummary: breathe sphinx-api $(SOURCE_DIR)/conf.py $(SOURCE_DIR)/index.rst
	@echo "** Building Sphinx autosummary documentation..."
	sphinx-build -M $(SPHINX_FORMAT) $(SOURCE_DIR) $(BUILD_DIR) \
		$(SPHINX_SUMMARY_OPTIONS) $(SPHINX_SUMMARY_EXTRA_OPTIONS) > $(SPHINX_SUMMARY_DEBUG) 2>&1
	@echo "** Sphinx autosummary documentation generated."
	@echo "** Saved output to $(SPHINX_SUMMARY_DEBUG)."
	@echo "** Warnings saved to $(SPHINX_SUMMARY_WARNINGS)."
	@count=$$(grep -E "WARNING|ERROR|CRITICAL" $(SPHINX_SUMMARY_DEBUG) | wc -l); \
	echo "** Found $$count warnings/errors in the Sphinx autosummary output."
	@echo

clean-sphinx-autosummary: check
	@echo "** Cleaning Sphinx autosummary-generated files..."
	rm -rf $(BUILD_DIR)
	rm -f $(SPHINX_SUMMARY_DEBUG)
	rm -f $(SPHINX_SUMMARY_WARNINGS)
	@echo

###############################################################################
# Sphinx docs build
################################################################################

SPHINX_DOCS_DEBUG = sphinx-docs-output.txt
SPHINX_DOCS_WARNINGS = sphinx-docs-warnings.txt
# Using more than one thread can cause issues interpreting the output
SPHINX_DOCS_OPTIONS = -j 1 -n -w $(SPHINX_DOCS_WARNINGS)
# Extra options for Sphinx command (can be specified in environment)
SPHINX_DOCS_EXTRA_OPTIONS ?=

sphinx-docs: sphinx-autosummary
	@echo "** Building Sphinx documentation..."
	sphinx-build -M $(SPHINX_FORMAT) $(SOURCE_DIR) $(BUILD_DIR) \
		$(SPHINX_DOCS_OPTIONS) $(SPHINX_DOCS_EXTRA_OPTIONS) > $(SPHINX_DOCS_DEBUG) 2>&1
	@echo "** Sphinx documentation generated."
	@echo "** Saved output to $(SPHINX_DOCS_DEBUG)."
	@echo "** Warnings saved to $(SPHINX_DOCS_WARNINGS)."
	@count=$$(grep -E "WARNING|ERROR|CRITICAL" $(SPHINX_DOCS_DEBUG) | wc -l); \
	echo "** Found $$count warnings/errors in the Sphinx documentation output."
	@echo

clean-sphinx-docs: check
	@echo "** Cleaning Sphinx documentation-generated files..."
	rm -rf $(BUILD_DIR)
	rm -f $(SPHINX_DOCS_DEBUG)
	rm -f $(SPHINX_DOCS_WARNINGS)
	@echo
