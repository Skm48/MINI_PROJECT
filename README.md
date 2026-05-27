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

## Modelling

### VGG16 Architecture and Training Strategy

#### 1. Architecture Overview

VGG16 is a 16-layer convolutional neural network characterised by its uniform architecture: all convolutional layers employ 3×3 filters with stride 1 and same padding, with max pooling (2×2, stride 2) applied after each convolutional block to progressively reduce spatial dimensions. The network comprises approximately 138 million parameters.

##### 1.1 Feature Extraction Backbone

The convolutional backbone consists of five sequential blocks, each applying a series of convolution–ReLU operations followed by max pooling. The feature maps decrease in spatial resolution while increasing in channel depth at each stage:

| Block | Conv Layers | Filters | Input Size | Output Size | Role |
|-------|------------|---------|------------|-------------|------|
| Block 1 | 2 | 64 | 224 × 224 × 3 | 112 × 112 × 64 | Low-level edge and texture detection |
| Block 2 | 2 | 128 | 112 × 112 × 64 | 56 × 56 × 128 | Corner and contour features |
| Block 3 | 3 | 256 | 56 × 56 × 128 | 28 × 28 × 256 | Mid-level structural patterns |
| Block 4 | 3 | 512 | 28 × 28 × 256 | 14 × 14 × 512 | High-level shape representations |
| Block 5 | 3 | 512 | 14 × 14 × 512 | 7 × 7 × 512 | Abstract semantic features |

The final feature map of shape 7 × 7 × 512 (25,088 values when flattened) serves as the input to the classification head.

##### 1.2 Custom Classification Head

The original VGG16 classifier was designed for ImageNet's 1,000-class task. For binary pneumonia classification, it was replaced with a lightweight fully connected architecture:

| Layer | Input Dim | Output Dim | Activation | Description |
|-------|-----------|------------|------------|-------------|
| Flatten | 512 × 7 × 7 | 25,088 | — | Spatial features reshaped to 1D |
| Dense | 25,088 | 256 | ReLU | Dimensionality reduction |
| Dropout | 256 | 256 | — | 50% dropout for regularisation |
| Dense | 256 | 2 | — | Output scores for Normal / Pneumonia |

Classification is performed by selecting the class with the higher output score (argmax). Weighted cross-entropy loss was employed during training to account for class imbalance (Normal:Pneumonia ≈ 1:2.9), with inverse-frequency weighting applied.

#### 2. Transfer Learning Strategy

The model was initialised with weights pretrained on ImageNet (ILSVRC-2012), a large-scale natural image dataset containing 1.2 million images across 1,000 categories. 
Transfer learning was employed on the rationale that low- and mid-level features learned from natural images (edges, textures, shapes) generalise well to medical imaging tasks.

##### 2.1 Phase 1 — Frozen Backbone Training

In the first training phase, all convolutional blocks (Blocks 1–5) were frozen, preserving the pretrained ImageNet weights. Only the custom classification head was optimised.

| Parameter | Value |
|-----------|-------|
| Trainable parameters | ~6.4M (classifier head only) |
| Frozen parameters | ~128M (conv blocks 1–5) |
| Optimiser | Adam |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Epochs | 10 |
| Batch size | 32 |

Phase 1 achieved 98% validation accuracy by epoch 5, demonstrating strong transferability of ImageNet features to chest X-ray classification.

##### 2.2 Phase 2 — Fine-Tuning Block 5

In the second phase, Block 5 (the final convolutional block) was unfrozen to allow domain-specific adaptation of high-level feature representations. A substantially reduced learning rate was used to prevent catastrophic forgetting of the pretrained features.

| Parameter | Value |
|-----------|-------|
| Unfrozen layers | Block 5 (3 conv layers) |
| Learning rate | 1 × 10⁻⁶ |
| All other hyperparameters | As Phase 1 |

**Outcome:** Fine-tuning did not yield improvement over Phase 1. Test accuracy decreased marginally (from 84.6% to 85.9% accuracy), with Normal class recall declining from 0.68 to 0.65. The Phase 1 frozen model was therefore retained as the final VGG16 model.

This result suggests that ImageNet features transfer sufficiently well to the chest X-ray domain that fine-tuning the final convolutional block provides no additional benefit for this dataset. 
Similar findings have been reported in prior medical imaging transfer learning studies (Tajbakhsh et al., 2016).


### Fusion Model Architecture and Training Strategy

#### 1. Fusion Approach Overview

This approach implements feature-level fusion, a multimodal ensemble technique that combines learned representations from multiple CNN architectures to produce a unified classification model.
Unlike score-level fusion (which averages or votes on final predictions), feature-level fusion concatenates the intermediate feature vectors extracted from each model's penultimate layer, preserving richer discriminative information prior to classification.
The rationale for fusion is that different CNN architectures learn complementary feature representations: VGG16 captures fine-grained textural patterns through its uniform 3×3 filter design, ResNet50 encodes hierarchical features via skip connections, and EfficientNet-B0 provides efficiently scaled representations through compound scaling.
Concatenating these diverse feature spaces enables the fusion classifier to exploit complementary strengths across architectures.

#### 2. Feature Extraction Pipeline

##### 2.1 Feature Sources

Features were extracted from the global average pooling (GAP) layer of each pretrained baseline model — the layer immediately preceding the classification head. All three backbone models were frozen during feature extraction (inference mode only).

| Model | Extraction Layer | Feature Dimension | Parameters |
|-------|-----------------|-------------------|------------|
| VGG16 | avgpool | 512 (after spatial averaging) | 138M |
| ResNet50 | avgpool | 2,048 | 25M |
| EfficientNet-B0 | avgpool | 1,280 | 5M |
| **Concatenated** | — | **3,840** | — |

For VGG16, the raw avgpool output is of shape 512 × 7 × 7. Spatial average pooling was applied to reduce this to a 512-dimensional vector, consistent with the dimensionality reduction applied by ResNet50 and EfficientNet-B0 internally. 
Initial experiments using the flattened 25,088-dimensional VGG16 features resulted in overfitting; the pooled 512-dimensional representation yielded superior fusion performance.

##### 2.2 Feature Normalisation

Each model's feature vectors occupy different numerical ranges due to differences in architecture and activation distributions. To prevent any single model's features from dominating the fused representation, per-model standardisation was applied using `sklearn.preprocessing.StandardScaler`:

- Scalers were fit on the training set features only
- The same fitted scalers were applied to validation and test sets (no data leakage)
- Post-normalisation, all feature dimensions have zero mean and unit variance

##### 2.3 Extraction Process

Features were extracted for all three dataset splits using forward hooks registered on each model's avgpool layer.
The extraction pipeline processes each image through all three frozen backbones in a single pass, storing the resulting feature vectors for subsequent fusion classifier training.

#### 3. Fusion Classifier Architecture

The fusion classifier is a fully connected neural network trained on the concatenated feature vectors. The architecture was designed to be lightweight relative to the backbone models, as the input features are already highly discriminative.

| Layer | Input Dim | Output Dim | Activation | Description |
|-------|-----------|------------|------------|-------------|
| Dense | 3,840 | 512 | ReLU | Feature compression |
| Dropout | 512 | 512 | — | 50% dropout for regularisation |
| Dense | 512 | 128 | ReLU | Further compression |
| Dropout | 128 | 128 | — | 30% dropout |
| Dense | 128 | 2 | — | Output scores for Normal / Pneumonia |

Total trainable parameters: approximately 2.0M (fusion head only; backbone weights are frozen).

#### 4. Training Configuration

The fusion classifier was trained on pre-extracted features, making training computationally inexpensive (no image processing required during optimisation).

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss function | Weighted CrossEntropyLoss (inverse-frequency) |
| Epochs | 20 |
| Batch size | 32 |

Training converged within approximately 15 epochs. The best model was selected based on minimum validation loss.


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
