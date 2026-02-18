"""
Generate polished PDF from the national Top 20 Medicaid fraud report.

Uses WeasyPrint for full CSS support (tables, page breaks, page numbers).
Requires: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /tmp/weasy_env/bin/python \
        src/experiments/generate_report_pdf.py
"""

import os
import re
import base64
from pathlib import Path

import markdown

# Must set DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib before running
import weasyprint

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
REPORT_MD = PROJECT_ROOT / "data" / "medicaid" / "national_top20_report.md"
REPORT_PDF = PROJECT_ROOT / "data" / "medicaid" / "national_top20_report.pdf"
REPORT_HTML = PROJECT_ROOT / "data" / "medicaid" / "national_top20_report_styled.html"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@page {
    size: letter;
    margin: 0.9in 0.85in 1in 0.85in;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #888;
        font-family: 'Helvetica', 'Arial', sans-serif;
    }
}

@page :first {
    @bottom-center { content: none; }
}

body {
    font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
}

/* ---- Title page ---- */
.title-page {
    page-break-after: always;
    text-align: center;
    padding-top: 2.5in;
}
.title-page h1 {
    font-size: 26pt;
    color: #0D47A1;
    border: none;
    margin-bottom: 10px;
    page-break-before: avoid;
}
.title-page .subtitle {
    font-size: 14pt;
    color: #555;
    margin-bottom: 30px;
}
.title-page .meta {
    font-size: 11pt;
    color: #777;
    margin-top: 40px;
}
.title-page .disclaimer {
    font-size: 9pt;
    color: #999;
    margin-top: 80px;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.4;
}

/* ---- Headings ---- */
h1 {
    font-size: 17pt;
    color: #0D47A1;
    border-bottom: 2.5px solid #0D47A1;
    padding-bottom: 4px;
    margin-top: 30px;
    margin-bottom: 12px;
    page-break-before: always;
}

h2 {
    font-size: 13pt;
    color: #1565C0;
    margin-top: 22px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    color: #1976D2;
    margin-top: 14px;
    margin-bottom: 4px;
}

/* ---- Body text ---- */
p {
    margin: 6px 0;
    orphans: 3;
    widows: 3;
}

strong {
    font-weight: 700;
    color: inherit;
}

em {
    font-style: italic;
}

/* ---- Lists ---- */
ul, ol {
    margin: 6px 0 6px 22px;
    padding: 0;
}
li {
    margin-bottom: 4px;
}

/* ---- Tables ---- */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    line-height: 1.35;
    table-layout: fixed;
    page-break-inside: auto;
}

th {
    background-color: #1565C0;
    color: white;
    font-weight: 600;
    padding: 7px 6px;
    text-align: left;
    border: 1px solid #1255A0;
    vertical-align: bottom;
}

td {
    padding: 5px 6px;
    border: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

tr:nth-child(even) td {
    background-color: #f7f9fc;
}

tr {
    page-break-inside: avoid;
}

/* ---- Code ---- */
code {
    background-color: #f0f2f5;
    padding: 1px 4px;
    font-size: 8.5pt;
    border-radius: 3px;
    font-family: 'Menlo', 'Consolas', monospace;
}

pre {
    background-color: #f0f2f5;
    padding: 10px 12px;
    font-size: 8pt;
    border-radius: 4px;
    overflow-x: hidden;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
    font-family: 'Menlo', 'Consolas', monospace;
    border: 1px solid #ddd;
}

pre code {
    background: none;
    padding: 0;
}

/* ---- Horizontal rules ---- */
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 18px 0;
}

/* ---- Images ---- */
img {
    max-width: 100%;
    display: block;
    margin: 12px auto;
}

.img-caption {
    text-align: center;
    font-size: 8.5pt;
    color: #666;
    margin-top: -6px;
    margin-bottom: 12px;
    font-style: italic;
}

/* ---- Avoid page breaks in bad places ---- */
h1, h2, h3 {
    page-break-after: avoid;
}

img {
    page-break-inside: avoid;
}

/* ---- Appendix styling ---- */
h1:last-of-type {
    /* Don't force page break for very last section */
}
"""


# ---------------------------------------------------------------------------
# Markdown to HTML conversion
# ---------------------------------------------------------------------------

def read_and_convert():
    """Read markdown, strip front matter, embed images, convert to HTML."""
    print(f"Reading: {REPORT_MD}")
    md_text = REPORT_MD.read_text(encoding="utf-8")

    # Strip YAML front matter
    md_text = re.sub(r"^---.*?---\s*", "", md_text, flags=re.DOTALL)

    # Strip LaTeX commands
    md_text = md_text.replace(r"\newpage", "")

    # Embed images as base64
    def embed_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)

        # Strip trailing { width=90% } etc
        img_path = re.sub(r"\{[^}]*\}", "", img_path).strip()

        # Resolve path: ../figures/... from data/medicaid/ -> figures/... from project root
        clean = re.sub(r"^\.\./", "", img_path)
        abs_path = PROJECT_ROOT / clean

        if abs_path.exists():
            data = base64.b64encode(abs_path.read_bytes()).decode("utf-8")
            ext = abs_path.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return (
                f'<img src="data:{mime};base64,{data}" alt="{alt_text}" />\n'
                f'<p class="img-caption">{alt_text}</p>'
            )
        else:
            print(f"  WARNING: image not found: {abs_path}")
            return f'<p class="img-caption">[Image not found: {img_path}]</p>'

    md_text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)(?:\{[^}]*\})?",
        embed_image,
        md_text,
    )

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    return html_body


def build_title_page():
    """Generate a styled cover/title page."""
    return """
    <div class="title-page">
        <h1>National Medicaid Fraud Investigation</h1>
        <div class="subtitle">Top 20 Targets</div>
        <div class="subtitle" style="font-size:12pt; color:#777;">
            Anomaly Detection in T-MSIS Provider Spending Data (2018 &ndash; 2024)
        </div>
        <div class="meta">
            Confidence Cartography Project<br/>
            February 2026
        </div>
        <div class="disclaimer">
            This report identifies statistically anomalous Medicaid billing patterns.
            Inclusion does not constitute an accusation of fraud.
            See the Methods section and Appendix for full context.
        </div>
    </div>
    """


def assemble_html(body_html: str) -> str:
    """Wrap HTML body with head, CSS, and title page."""
    title_page = build_title_page()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
{CSS}
</style>
</head>
<body>
{title_page}
{body_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(html: str):
    """Render HTML to PDF via WeasyPrint."""
    print(f"Generating PDF via WeasyPrint...")

    # Save styled HTML for debugging
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"  Saved HTML: {REPORT_HTML}")

    # Generate PDF
    doc = weasyprint.HTML(string=html)
    doc.write_pdf(str(REPORT_PDF))

    size_mb = REPORT_PDF.stat().st_size / 1024 / 1024
    print(f"  Saved PDF:  {REPORT_PDF}")
    print(f"  Size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NATIONAL TOP-20 REPORT: PDF GENERATION")
    print("=" * 60)

    body_html = read_and_convert()

    # Count embedded images
    n_images = body_html.count("data:image/")
    print(f"  Images embedded: {n_images}")

    full_html = assemble_html(body_html)
    print(f"  HTML size: {len(full_html):,} chars")

    generate_pdf(full_html)

    print("\nDone!")


if __name__ == "__main__":
    main()
