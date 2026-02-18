# arXiv Submission Guide

## Where to submit

**Primary:** arXiv — https://arxiv.org/submit
**Category:** cs.CL (Computation and Language) — primary
**Cross-list:** cs.AI, cs.LG (optional but helps visibility)

arXiv is free, takes 1–2 business days to appear, and gives you a permanent timestamp and DOI-like identifier (e.g. arXiv:2502.XXXXX). This is standard in ML/NLP — most people read papers here before they ever appear in a journal.

---

## Before submitting: checklist

### 1. Convert the paper to PDF
The paper is currently in Markdown at `paper/confidence_cartography.md`. arXiv accepts PDF directly. Options:

**Option A (easiest) — Pandoc:**
```bash
pandoc confidence_cartography.md -o confidence_cartography.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

**Option B — LaTeX (better formatting):**
Convert the Markdown to a .tex file and use the standard NeurIPS or ACL template. The NeurIPS 2024 template is appropriate for this type of paper:
- Download from: https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
- Paste the paper content into the template
- Compile with `pdflatex`

**Option C — Google Docs / Word:**
Paste into Google Docs, format manually, export to PDF. Lowest effort but less clean typesetting.

### 2. Prepare figures
Select 4–5 key figures from `figures/` to include in the paper. Recommended:
- `figures/mandela/calibration/calibration_scatter_6.9b.png` — the main ρ=0.652 result (Figure 1)
- `figures/scaling/a1_metrics_scaling.png` — accuracy vs. model size (Figure 2)
- `figures/mandela/calibration/calibration_items_6.9b.png` — per-item comparison (Figure 3)
- `figures/scaling/medical_scaling.png` — medical generalization (Figure 4)
- `figures/targeted_resampling/compute_accuracy_tradeoff.png` — resampling efficiency (Figure 5)

### 3. Fill in the appendices
The paper has placeholder appendices (A, B, C). Before submitting, either:
- Fill them in with the actual content (preferred), or
- Remove them if you want to submit quickly — the main paper is complete without them

### 4. Author information
You'll need to add your name and affiliation to the paper before submitting. arXiv requires at least one author with a verified email.

---

## Submission steps

1. **Create an arXiv account** at https://arxiv.org/register if you don't have one
   - Needs institutional email OR endorsement from an existing arXiv author
   - If you don't have an institutional email, you need one endorsement — email someone in NLP/ML who has published on arXiv and ask them to endorse you (takes 5 minutes for them)

2. **Start a new submission** at https://arxiv.org/submit
   - Select: Computer Science → Computation and Language (cs.CL)
   - Cross-list: cs.AI, cs.LG

3. **Upload files**
   - If submitting PDF only: just upload the PDF
   - If submitting LaTeX source: zip the .tex + figures and upload the zip

4. **Fill in metadata**
   - Title: "Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models"
   - Authors: [your name]
   - Abstract: copy the abstract from the paper (plain text, no markdown)
   - MSC Class: leave blank
   - ACM Class: leave blank
   - Report number: leave blank

5. **Submit** — goes to moderation, typically appears the next business day

---

## After submission

- You'll get an arXiv ID (e.g. arXiv:2502.12345) within 24 hours
- Share the link — this is sufficient to establish priority and get feedback
- You can update the paper any time by submitting a new version (v2, v3, etc.) — the original submission timestamp is preserved

---

## Optional next steps (not required)

If you want actual peer review later:
- **ACL Rolling Review (ARR):** https://aclrollingreview.org — monthly submission deadlines, routes to ACL/EMNLP/NAACL
- **EMNLP 2026:** deadline typically ~May/June
- **Workshop track:** Consider the BlackboxNLP workshop at EMNLP — directly relevant venue for this type of interpretability work

None of this is necessary for the preprint to be useful and visible. Most of the people who would care about this work will find it on arXiv.

---

## Title / abstract tweaks (optional)

The current title is accurate and descriptive. Alternative framings if you want to adjust tone:
- "What Language Models Believe: Confidence as a Cultural Consensus Sensor"
- "Teacher-Forced Confidence Tracks Human False-Belief Prevalence in Language Models"
- Keep current — "Confidence Cartography" is a clean, memorable phrase

The abstract is complete. If you want to trim it, the minimum viable abstract is everything through "…3–5× lower compute cost."
