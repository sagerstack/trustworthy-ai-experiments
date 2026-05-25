# Trustworthy AI Experiments

Adversarial machine learning experiments for SUTD's Trustworthy AI course
(Week 1, Lecture 2). The current experiment generates **FGSM evasion attacks**
against an image classifier and studies how an accurate, confident model
collapses under near-imperceptible perturbations.

## FGSM Attack on MNIST

A ResNet-50 (ImageNet-pretrained) is fine-tuned on MNIST, then attacked with the
**Fast Gradient Sign Method** via IBM's [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

### Notebooks (`fgsm-attack/`)

| Notebook | Purpose |
|---|---|
| `mnist-resnet50.ipynb` | Train: fine-tune ResNet-50 on MNIST, evaluate clean accuracy, save the checkpoint (`resnet50_mnist.pt`). |
| `mnist-fgsm-attack.ipynb` | Attack: load the checkpoint, wrap it in ART, run FGSM across an ε sweep, and analyze degradation. |

### Approach

- **Model**: ResNet-50, ImageNet-pretrained. Final layer replaced with `Linear(2048, 10)`;
  only `layer4` + head fine-tuned (~15M of 23.5M params). Pretrained `conv1` kept; MNIST
  grayscale expanded to 3 channels via `Grayscale(3)`.
- **Normalization inside the model**: ART receives `[0,1]` images with `clip_values=(0,1)`,
  so ε maps to a real pixel-space budget and adversarial images stay valid.
- **Attack**: `FastGradientMethod` (untargeted, L∞). ε sweep `[0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]`.
  Single-step FGSM is ε-independent in its gradient sign, so one backward pass is reused across all ε.

### Key findings

| Metric | Result |
|---|---|
| Clean test accuracy (10k) | **96.49%** |
| Train accuracy / generalization gap | 99.25% / +2.76 pp (strong generalization) |
| Adversarial accuracy @ ε≈0.031 (8/255) | **~9.7%** (≈ chance) |
| Curve floor | ~8% reached at **ε≈0.03**, flat through 0.30 |

- A **near-imperceptible** perturbation (ε≈0.03, ~8/255) collapses the model to chance.
- Per-class robustness is **uneven and non-monotonic**: the attack herds predictions into
  a few attractor classes (digits 5, 7), inflating their recall while precision collapses.
- **Confidence is not a usable safety signal**: under attack the model stays 74–100% confident,
  and the correct-vs-incorrect confidence gap shrinks to ~0 (even negative past ε=0.05) —
  the model is *confidently wrong*.

## Setup

Managed with [Poetry](https://python-poetry.org/) (Python 3.13).

```bash
poetry install                  # install dependencies into .venv
poetry run jupyter lab          # or select the .venv kernel in VS Code
```

Dependencies: `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`,
`adversarial-robustness-toolbox`.

## Notes

- The model checkpoint (`*.pt`, ~94 MB) and downloaded MNIST data are git-ignored —
  regenerate the checkpoint by running `mnist-resnet50.ipynb` end-to-end.
- The FGSM attack runs on **CPU** (ART has no Apple MPS backend); training uses MPS/CUDA if available.
