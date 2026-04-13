# Hybrid CNN Fusion for Pneumonia Detection

Multi-architecture deep learning pipeline for pneumonia detection from chest X-rays, with feature-level fusion and Grad-CAM explainability analysis.

## Overview

This project evaluates three CNN architectures (VGG16, ResNet50, EfficientNet-B0) for binary pneumonia classification using transfer learning, develops a hybrid model via feature-level concatenation, and applies Grad-CAM to compare attention patterns across architectures.

## Project structure

```
hybrid-cnn-pneumonia/
├── configs/
│   └── config.yaml          # All hyperparameters + paths
├── data/
│   └── chest_xray/           # Kaggle dataset (not tracked)
├── models/
│   └── checkpoints/           # Saved .pth files (not tracked)
├── notebooks/
│   ├── 01_setup_eda.ipynb     # Data pipeline + EDA
│   ├── 02_baselines.ipynb     # VGG16, ResNet50, EfficientNet
│   ├── 03_fusion.ipynb        # Feature-level fusion
│   └── 04_gradcam.ipynb       # Explainability analysis
├── outputs/
│   ├── figures/               # Training curves, comparison charts
│   ├── gradcam/               # Heatmap visualisations
│   └── metrics/               # JSON metric logs
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Data loading + preprocessing
│   ├── models.py              # Baseline CNN architectures
│   ├── train.py               # Training loop + MLflow logging
│   ├── evaluate.py            # Metrics + confusion matrix
│   ├── fusion.py              # Feature extraction + fusion head
│   ├── gradcam.py             # Grad-CAM generation
│   └── utils.py               # Config, seeds, device, helpers
├── mlruns/                    # MLflow tracking (not tracked)
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al. (2018). 5,863 labelled anterior-posterior chest X-ray images from paediatric patients.

The default validation split (16 images) is merged with training data and re-split into 80/10/10 stratified partitions.

## Quick start (Google Colab)

1. Clone this repo
2. Upload `kaggle.json` to Colab (or copy dataset from Drive)
3. Open `notebooks/01_setup_eda.ipynb` and run all cells
4. Proceed through notebooks 02 → 03 → 04 in order

## Tech stack

- **Framework:** PyTorch + torchvision
- **Models:** VGG16, ResNet50, EfficientNet-B0 (ImageNet pretrained)
- **Explainability:** pytorch-grad-cam
- **Tracking:** MLflow
- **Evaluation:** scikit-learn

## Results

*To be updated after training.*

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| VGG16 | — | — | — | — | — |
| ResNet50 | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — |
| **Fusion** | — | — | — | — | — |

## References

1. Kermany et al. (2018) — Cell, 172(5), 1122–1131
2. Simonyan & Zisserman (2015) — VGG, ICLR 2015
3. He et al. (2016) — ResNet, CVPR 2016
4. Tan & Le (2019) — EfficientNet, ICML 2019
5. Selvaraju et al. (2017) — Grad-CAM, ICCV 2017

## License

MIT
