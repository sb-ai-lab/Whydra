from pathlib import Path

from pygments.lexers.special import TextLexer
from sphinx import addnodes

root = Path(__file__).resolve().parents[2]

project = "Whydra"
copyright = "2026, Skoltech"
release = (root / "whydra" / "VERSION").read_text(encoding="ascii").strip()
language = "en"

extensions = ["myst_parser", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
}


def _link_contributor_download(app, doctree):
    # MyST can classify Markdown files outside the Sphinx source tree as pages.
    contributor_doc = app.env.path2doc(str(root / "CONTRIBUTORS.md"))
    for link in list(doctree.findall(addnodes.pending_xref)):
        if link.get("refdomain") == "doc" and link["reftarget"] == contributor_doc:
            download = addnodes.download_reference(
                "", "", *link.children, reftarget="/../../CONTRIBUTORS.md"
            )
            download.source, download.line = link.source, link.line
            link.replace_self(download)


def _keep_downloads_out_of_toc(app, doctree):
    # Keep file names as text in the TOC and as links in the page headings.
    toc = app.env.tocs.get(app.env.docname)
    if toc is not None:
        for link in list(toc.findall(addnodes.download_reference)):
            link.replace_self(link.children)


def setup(app):
    """Register the Mermaid lexer and adjust download links in the documentation."""
    app.add_lexer("mermaid", TextLexer)
    app.connect("doctree-read", _link_contributor_download, priority=499)
    app.connect("doctree-read", _keep_downloads_out_of_toc, priority=501)
