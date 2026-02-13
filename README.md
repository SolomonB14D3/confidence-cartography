# Confidence Cartography

**Using Language Models as Sensors for the Structure of Human Knowledge**

## Overview

This project treats language model confidence patterns as a scientific instrument — not asking "is the model right?" but "what does the model's uncertainty reveal about the structure of human knowledge?"

Key findings:
- Model confidence distinguishes true from false claims with 90% accuracy at 6.9B scale
- Confidence measures **transmissibility** (how often something is repeated in text), not truth directly
- These usually correlate, but diverge predictably for cultural misconceptions (Mandela Effects)
- Model confidence ratios significantly correlate with measured human false belief prevalence (ρ=0.652, p=0.016)
- The sensor generalizes to medical domain claims (88% at 6.9B, p=0.01)

## Project Structure

```
src/
├── engine.py              # Core confidence extraction
├── schema.py              # Data classes and storage
├── viz.py                 # Visualization toolkit
├── training_utils.py      # Fine-tuning helpers
├── scaling_utils.py       # Multi-model loading
└── experiments/           # All experiment scripts

data/
├── prompts/               # Input prompt sets
└── results/               # JSONL outputs from all experiments

figures/                   # All generated visualizations (~170 figures)
```

## Model

All experiments use the Pythia model family (EleutherAI): 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B parameters.

## Requirements

```
torch
transformers
datasets
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

## Hardware

Developed on Apple M3 Ultra with 96GB unified memory.
