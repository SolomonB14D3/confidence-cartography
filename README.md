# Confidence Cartography

**Using Language Models as Sensors for the Structure of Human Knowledge**

A paper-ready ML interpretability study showing that teacher-forced confidence (the probability a causal LM assigns to its own tokens) is a practical false-belief sensor that generalizes across model scales and domains.

> Paper: [DOI: 10.5281/zenodo.18703506](https://doi.org/10.5281/zenodo.18703506)

---

## Key Findings

- Model confidence ratios significantly correlate with human false-belief prevalence (**ρ=0.652, p=0.016**) across Pythia 160M→12B
- Distinguishes true from false factual claims with **90% accuracy** at 6.9B scale
- Generalizes to out-of-domain medical claims (**88% accuracy** at 6.9B, p=0.01)
- "Mandela effect" false memories show systematically lower confidence than accurate memories — model confidence tracks transmissibility, not truth directly
- Targeted resampling at low-confidence token positions outperforms uniform best-of-N at lower compute cost

---

## Project Structure

```
src/
├── engine.py                         # Core confidence extraction (teacher-forced + generation)
├── schema.py                         # Data classes and storage
├── viz.py                            # Visualization toolkit
├── training_utils.py                 # Fine-tuning helpers
├── scaling.py / scaling_viz.py       # Multi-model loading and scale plots
└── experiments/
    ├── exp1_baselines.py             # Baseline confidence by claim category
    ├── exp2_truth.py                 # True vs. false discrimination
    ├── exp3_contested.py             # Contested/controversial claims
    ├── exp4_training.py              # Training dynamics
    ├── exp5_consensus.py             # Consensus vs. minority views
    ├── exp6_anomaly.py               # Anomaly detection framing
    ├── exp9_medical_validation.py    # OOD medical generalization
    ├── exp_a1–exp_a4_*.py           # Scaling experiments (160M→12B)
    ├── exp_b9_rlhf_regime.py        # RLHF vs. base model comparison
    ├── exp_b14_checkpoint_stability.py  # Checkpoint stability (null result)
    ├── exp_b15_text_properties.py   # Text surface property controls
    ├── exp_c_mandela.py             # Mandela effect false-memory calibration
    ├── exp_mandela_*.py             # Mandela expanded + reanalysis
    ├── exp_targeted_resampling*.py  # Targeted resampling at uncertainty points
    └── exp_token_localization*.py   # Token-level uncertainty localization

data/
├── prompts/                          # Input prompt sets
└── results/                          # JSONL outputs from all experiments (~252 files, 20 MB)

figures/                              # All generated visualizations (~170 figures)

paper/
├── confidence_cartography.md         # Paper draft (Markdown)
├── confidence_cartography.pdf        # Compiled PDF
└── paper.css                         # Stylesheet for HTML render
```

---

## Reproducing Experiments

### Requirements

```
pip install -r requirements.txt
```

### Running an experiment

```bash
python src/experiments/exp1_baselines.py
python src/experiments/exp2_truth.py
# etc.
```

Results are written as JSONL to `data/results/`. Figures are written to `figures/`.

### Core engine

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

```
@misc{sanchez2026confidence,
  author  = {Sanchez, Bryan},
  title   = {Confidence Cartography: Teacher-Forced Probability as a False-Belief Sensor in Language Models},
  year    = {2026},
  doi     = {10.5281/zenodo.18703506},
  url     = {https://doi.org/10.5281/zenodo.18703506}
}
```
