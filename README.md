# Confidence Cartography

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718611.svg)](https://doi.org/10.5281/zenodo.18718611)

**Using Language Models as Sensors for the Structure of Human Knowledge**

A paper-ready ML interpretability study showing that teacher-forced confidence (the probability a causal LM assigns to its own tokens) is a practical false-belief sensor that generalizes across model scales and domains.

> Paper: [DOI: 10.5281/zenodo.18718611](https://doi.org/10.5281/zenodo.18718611)

> **Pip-installable toolkit:** [`confidence-cartography-toolkit`](https://github.com/SolomonB14D3/confidence-cartography-toolkit) — reproduce the key results in 3 lines of Python:
> ```python
> import confidence_cartography as cc
> results = cc.evaluate_mandela_effect("EleutherAI/pythia-6.9b")
> print(f"Spearman rho: {results.rho:.3f}, p={results.p_value:.4f}")
> ```

---

## How It Works

Teacher-forced probability extraction measures the confidence a language model assigns to each token in a given text. By comparing confidence on true vs. false versions of claims, we detect systematic biases that mirror human false beliefs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE CARTOGRAPHY                           │
│                   False-Belief Sensor Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUT: Claim Pair                                                 │
│   ┌─────────────────────┐    ┌─────────────────────┐               │
│   │ "Luke, I am your    │    │ "No, I am your      │               │
│   │  father." (popular) │    │  father." (correct) │               │
│   └──────────┬──────────┘    └──────────┬──────────┘               │
│              │                          │                           │
│              ▼                          ▼                           │
│   ┌──────────────────────────────────────────────────┐             │
│   │           PYTHIA MODEL (160M → 12B)              │             │
│   │        Teacher-Forced Probability Mode           │             │
│   └──────────────────────────────────────────────────┘             │
│              │                          │                           │
│              ▼                          ▼                           │
│        P(popular) = 0.076         P(correct) = 0.052               │
│                                                                     │
│              └──────────┬───────────────┘                          │
│                         ▼                                           │
│   ┌──────────────────────────────────────────────────┐             │
│   │  Confidence Ratio = P(popular) / P(correct)      │             │
│   │                   = 1.46 → MODEL PREFERS WRONG   │             │
│   └──────────────────────────────────────────────────┘             │
│                         │                                           │
│                         ▼                                           │
│   ┌──────────────────────────────────────────────────┐             │
│   │  Correlate with Human False-Belief Prevalence    │             │
│   │  ρ = 0.652 (p = 0.016) at 6.9B parameters        │             │
│   └──────────────────────────────────────────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Findings

- Model confidence ratios significantly correlate with human false-belief prevalence (**ρ=0.652, p=0.016**) across Pythia 160M→12B
- Distinguishes true from false factual claims with **90% accuracy** at 6.9B scale
- Generalizes to out-of-domain medical claims (**88% accuracy** at 6.9B, p=0.01)
- "Mandela effect" false memories show systematically lower confidence than accurate memories — model confidence tracks transmissibility, not truth directly
- Targeted resampling at low-confidence token positions outperforms uniform best-of-N at lower compute cost

### Highlighted Results

| Finding | Value | Significance |
|---------|-------|--------------|
| **1B Parameter Peak** | ρ = 0.718 | Strongest correlation occurs at mid-scale, not largest model |
| **Early Emergence** | Step 256 | Signal is stable early in training—useful for studying training dynamics |
| **Compute Efficiency** | 3-5x | Targeted resampling achieves comparable improvement at lower cost vs. uniform best-of-N |

---

## Testing the Persistence of False Beliefs

This repository provides the baseline for how **Pythia (2023)** encoded cultural misconceptions. We invite researchers to run these extraction scripts on newer models (e.g., Llama 3, Qwen 2.5, Gemma 2) to answer:

- **Do false beliefs persist?** Are "Mandela Effects" still encoded in 2025/2026 models?
- **Do they fade?** As training corpora evolve, do old misconceptions weaken?
- **Do new ones emerge?** What false beliefs are uniquely strong in newer models?

To test a new model, modify `src/scaling.py` to add your model to `MODEL_REGISTRY`, then run:

```bash
python src/experiments/exp_c_mandela.py --all
```

---

## Project Structure

```
confidence-cartography/
├── data/
│   ├── mandela_effect.json           # 13 Mandela Effect items (popular vs correct)
│   ├── medical_claims.json           # 25 medical claim pairs (true vs false)
│   └── results/                       # JSONL outputs from all experiments
│
├── notebooks/
│   └── reproduce_figure1.ipynb       # Reproduce the ρ=0.652 correlation plot
│
├── src/
│   ├── engine.py                     # Core confidence extraction
│   ├── schema.py                     # Data classes and storage
│   ├── scaling.py / scaling_viz.py   # Multi-model loading and scale plots
│   └── experiments/
│       ├── exp_c_mandela.py          # Mandela effect analysis
│       ├── exp9_medical_validation.py # Medical domain generalization
│       ├── exp_targeted_resampling*.py # Compute-efficient resampling
│       └── ...                        # Additional experiments
│
├── figures/                           # All generated visualizations (~170 figures)
├── paper/                             # Paper draft (Markdown, HTML, PDF)
└── requirements.txt                   # Dependencies with version pins
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Reproduce Figure 1

```bash
jupyter notebook notebooks/reproduce_figure1.ipynb
```

Or run the Mandela experiment directly:

```bash
python src/experiments/exp_c_mandela.py --all
```

### Core Engine

`src/engine.py` exposes two modes:

- **Teacher-forced mode**: score a fixed completion — measure the model's log-probability of each token
- **Generation mode**: sample completions and score them

---

## Models

All core experiments use the [Pythia](https://github.com/EleutherAI/pythia) model family (EleutherAI): 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B parameters. Cross-architecture validation uses Qwen2.5 7B and 32B.

---

## Hardware

Developed on Apple M3 Ultra with 96 GB unified memory. Experiments are CPU/MPS compatible; GPU (CUDA) will be faster for the larger models.

---

## Citation

```bibtex
@misc{sanchez2026confidence,
  author  = {Sanchez, Bryan},
  title   = {Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models},
  year    = {2026},
  doi     = {10.5281/zenodo.18718611},
  url     = {https://doi.org/10.5281/zenodo.18718611}
}
```
