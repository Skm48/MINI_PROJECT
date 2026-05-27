# VGG16 Architecture and Training Strategy

## 1. Architecture Overview

VGG16 (Simonyan & Zisserman, 2015) is a 16-layer convolutional neural network characterised by its uniform architecture: all convolutional layers employ 3×3 filters with stride 1 and same padding, with max pooling (2×2, stride 2) applied after each convolutional block to progressively reduce spatial dimensions. The network comprises approximately 138 million parameters.

### 1.1 Feature Extraction Backbone

The convolutional backbone consists of five sequential blocks, each applying a series of convolution–ReLU operations followed by max pooling. The feature maps decrease in spatial resolution while increasing in channel depth at each stage:

| Block | Conv Layers | Filters | Input Size | Output Size | Role |
|-------|------------|---------|------------|-------------|------|
| Block 1 | 2 | 64 | 224 × 224 × 3 | 112 × 112 × 64 | Low-level edge and texture detection |
| Block 2 | 2 | 128 | 112 × 112 × 64 | 56 × 56 × 128 | Corner and contour features |
| Block 3 | 3 | 256 | 56 × 56 × 128 | 28 × 28 × 256 | Mid-level structural patterns |
| Block 4 | 3 | 512 | 28 × 28 × 256 | 14 × 14 × 512 | High-level shape representations |
| Block 5 | 3 | 512 | 14 × 14 × 512 | 7 × 7 × 512 | Abstract semantic features |

The final feature map of shape 7 × 7 × 512 (25,088 values when flattened) serves as the input to the classification head.

### 1.2 Custom Classification Head

The original VGG16 classifier was designed for ImageNet's 1,000-class task. For binary pneumonia classification, it was replaced with a lightweight fully connected architecture:

| Layer | Input Dim | Output Dim | Activation | Description |
|-------|-----------|------------|------------|-------------|
| Flatten | 512 × 7 × 7 | 25,088 | — | Spatial features reshaped to 1D |
| Dense | 25,088 | 256 | ReLU | Dimensionality reduction |
| Dropout | 256 | 256 | — | 50% dropout for regularisation |
| Dense | 256 | 2 | — | Output scores for Normal / Pneumonia |

Classification is performed by selecting the class with the higher output score (argmax). Weighted cross-entropy loss was employed during training to account for class imbalance (Normal:Pneumonia ≈ 1:2.9), with inverse-frequency weighting applied.

## 2. Transfer Learning Strategy

The model was initialised with weights pretrained on ImageNet (ILSVRC-2012), a large-scale natural image dataset containing 1.2 million images across 1,000 categories. Transfer learning was employed on the rationale that low- and mid-level features learned from natural images (edges, textures, shapes) generalise well to medical imaging tasks (Raghu et al., 2019).

### 2.1 Phase 1 — Frozen Backbone Training

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

### 2.2 Phase 2 — Fine-Tuning Block 5

In the second phase, Block 5 (the final convolutional block) was unfrozen to allow domain-specific adaptation of high-level feature representations. A substantially reduced learning rate was used to prevent catastrophic forgetting of the pretrained features.

| Parameter | Value |
|-----------|-------|
| Unfrozen layers | Block 5 (3 conv layers) |
| Learning rate | 1 × 10⁻⁶ |
| All other hyperparameters | As Phase 1 |

**Outcome:** Fine-tuning did not yield improvement over Phase 1. Test accuracy decreased marginally (from 84.6% to 85.9% accuracy), with Normal class recall declining from 0.68 to 0.65. The Phase 1 frozen model was therefore retained as the final VGG16 model.

This result suggests that ImageNet features transfer sufficiently well to the chest X-ray domain that fine-tuning the final convolutional block provides no additional benefit for this dataset. Similar findings have been reported in prior medical imaging transfer learning studies (Tajbakhsh et al., 2016).

## 3. Results

### 3.1 Test Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.846 |
| Weighted F1 | 0.836 |
| AUC-ROC | 0.963 |

### 3.2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.97 | 0.61 | 0.75 | 234 |
| Pneumonia | 0.81 | 0.99 | 0.89 | 390 |

The model exhibits high pneumonia recall (0.99) but substantially lower Normal recall (0.61), indicating a bias towards positive (pneumonia) predictions. This imbalance is attributable to the skewed class distribution in the training data and the model's tendency to favour the majority class.

## 4. Grad-CAM Explainability Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) was applied to the final convolutional layer (`features[28]`, ReLU activation in Block 5) to visualise the spatial regions contributing to the model's predictions.

**Key findings:**

- **Inconsistent localisation:** VGG16's attention patterns vary considerably across test images. In some cases, the model correctly highlights lung parenchyma regions exhibiting consolidation or opacity. In other cases, attention is concentrated on peripheral structures (shoulder area, upper chest border, mediastinum).

- **Evidence of shortcut learning:** The model occasionally bases predictions on imaging artefacts or patient positioning cues rather than clinically relevant lung pathology. This phenomenon, termed shortcut learning (Geirhos et al., 2020), is a recognised limitation of CNNs trained on medical imaging datasets where confounding variables (patient positioning, image acquisition parameters) may correlate with diagnostic labels.

- **False negative analysis:** Grad-CAM heatmaps for misclassified pneumonia cases (false negatives) reveal attention directed towards the cardiac silhouette and mediastinum rather than the lung fields, suggesting the model fails to detect subtle or diffuse pneumonia patterns.

## 5. References

- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
- Raghu, M., et al. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *NeurIPS 2019*.
- Tajbakhsh, N., et al. (2016). Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning? *IEEE TMI*, 35(5), 1299–1312.
- Geirhos, R., et al. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, 2, 665–673.
