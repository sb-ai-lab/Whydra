# Build and preview the documentation locally.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html clean livehtml linkcheck

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Documentation built in $(BUILDDIR)/html"

livehtml:
	sphinx-autobuild --host 0.0.0.0 --port 8080 "$(SOURCEDIR)" "$(BUILDDIR)/html"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"
	@echo "Removed $(BUILDDIR)"
