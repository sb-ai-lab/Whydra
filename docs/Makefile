# Sphinx configuration
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: html clean

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html"

clean:
	@rm -rf "$(BUILDDIR)"
	@echo "Cleaned $(BUILDDIR)"

# live-reload для локальной разработки
livehtml:
	sphinx-autobuild --host 0.0.0.0 --port 8080 "$(SOURCEDIR)" "$(BUILDDIR)/html"

# проверка битых ссылок
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)