"""Convert report.md to a well-formatted PDF (markdown-it-py + WeasyPrint)."""
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from weasyprint import HTML

src = Path(sys.argv[1])
out = Path(sys.argv[2])
text = src.read_text(encoding="utf-8")
# GFM task items -> inline ballot-box glyphs (keeps the box on the text line).
text = re.sub(r"^(\s*[-*+] )\[[xX]\] ", r"\1☑ ", text, flags=re.M)
text = re.sub(r"^(\s*[-*+] )\[ \] ", r"\1☐ ", text, flags=re.M)

md = (
    MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False})
    .enable("table")
    .enable("strikethrough")
    .use(anchors_plugin, min_level=1, max_level=4,
         slug_func=lambda s: re.sub(r"[^\w\- ]", "", s.strip().lower()).replace(" ", "-"))
)
body = md.render(text)

# Mark tables by column count so CSS can tighten wide ones.
def tag_table(m):
    tbl = m.group(0)
    head = re.search(r"<tr>(.*?)</tr>", tbl, re.S)
    ncols = len(re.findall(r"<t[hd][ >]", head.group(1))) if head else 0
    cls = "wide" if ncols >= 5 else ("kv" if ncols == 2 else "")
    # Header row with all-empty cells (| | |) -> drop it.
    thead = re.search(r"<thead>(.*?)</thead>", tbl, re.S)
    if thead and not re.sub(r"<[^>]+>|\s", "", thead.group(1)):
        tbl = tbl.replace(thead.group(0), "")
        cls += " nohead"
    return tbl.replace("<table>", f'<table class="{cls.strip()} c{ncols}">', 1)

body = re.sub(r"<table>.*?</table>", tag_table, body, flags=re.S)

title = re.search(r"^# (.+)$", text, re.M).group(1)

css = r"""
@page {
  size: A4;
  margin: 20mm 17mm 20mm 17mm;
  @bottom-center { content: counter(page) " / " counter(pages);
                   font: 8.5pt 'DejaVu Sans'; color: #777; }
  @top-right { content: "PCM write-absorbing tier — report";
               font: 8pt 'DejaVu Sans'; color: #999; }
}
@page :first { @top-right { content: none; } }
html { font-family: 'DejaVu Serif', serif; font-size: 10pt; line-height: 1.45;
       color: #1a1a1a; }
body { margin: 0; }
p, li { text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
p { margin: 0 0 0.6em 0; }
ul, ol { margin: 0.2em 0 0.7em 0; padding-left: 1.4em; }
li { margin: 0.15em 0; }
li > p { margin: 0.2em 0; }
h1, h2, h3, h4 { font-family: 'DejaVu Sans', sans-serif; line-height: 1.25;
                 page-break-after: avoid; break-after: avoid; text-align: left; }
h1 { font-size: 17pt; color: #0b3d63; border-bottom: 2px solid #0b3d63;
     padding-bottom: 4px; margin: 0 0 0.7em 0; }
h1:not(:first-of-type) { page-break-before: always; }
h2 { font-size: 13pt; color: #0b3d63; margin: 1.4em 0 0.5em 0;
     border-bottom: 1px solid #c8d3dd; padding-bottom: 2px; }
h3 { font-size: 11pt; color: #1f4e79; margin: 1.1em 0 0.4em 0; }
h4 { font-size: 10pt; margin: 0.9em 0 0.3em 0; }
a { color: #1f4e79; text-decoration: none; }
strong { color: #000; }
hr { border: none; border-top: 1px solid #bbb; margin: 1.2em 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 0.86em;
       background: #f2f4f6; padding: 0 2px; border-radius: 2px;
       overflow-wrap: anywhere; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; line-height: 1.35;
      background: #f6f8fa; border: 1px solid #dde2e6; border-left: 3px solid #1f4e79;
      padding: 7px 9px; white-space: pre-wrap; overflow-wrap: anywhere;
      page-break-inside: avoid; margin: 0.4em 0 0.9em 0; }
pre code { background: none; padding: 0; font-size: 1em; }
blockquote { margin: 0.6em 0; padding: 0.3em 0.9em; border-left: 3px solid #c8d3dd;
             color: #333; background: #fafbfc; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0 1em 0;
        font-family: 'DejaVu Sans', sans-serif; font-size: 8.3pt; line-height: 1.3;
        page-break-inside: auto; }
tr { page-break-inside: avoid; break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid #c9d1d9; padding: 3px 5px; vertical-align: top;
         text-align: left; overflow-wrap: break-word; hyphens: manual; }
td:first-child, th:first-child { min-width: 2.2em; }
th { background: #e8eef4; font-weight: bold; color: #0b3d63; }
tbody tr:nth-child(even) td { background: #fafbfc; }
td code, th code { font-size: 0.92em; overflow-wrap: anywhere; }
table.wide { font-size: 7.4pt; }
table.wide th, table.wide td { padding: 2px 4px; }
table.c7, table.c8 { font-size: 6.9pt; }
table.kv td:first-child { width: 24%; font-weight: bold; background: #f2f5f8; }
table.nohead tr:first-child td { border-top: 1px solid #c9d1d9; }
li.task-list-item { list-style: none; margin-left: -1.2em; }
li.task-list-item input { margin: 0 0.4em 0 0; vertical-align: -1px; }
td[style*="right"], th[style*="right"] { text-align: right; }
"""

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>{body}</body></html>"""

HTML(string=html, base_url=str(src.parent)).write_pdf(out)
print("wrote", out)
