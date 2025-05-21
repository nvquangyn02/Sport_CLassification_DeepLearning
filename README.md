# 🏅 Sports Image Classification Project

**Updated:** May 21, 2025

---

## 📖 Overview

This project builds a deep learning model to classify images of **100 different sports** using **transfer learning** and **fine-tuning**. It aims to achieve high classification accuracy (initially ~85%, improved to **95–98%**) on a custom dataset using models like MobileNetV2 and EfficientNetB0.

---

## 🎯 Objectives

- Classify 100 sports categories from images with high accuracy.
- Visualize results using sample images, confusion matrices, and training plots.
- Optimize training time and improve performance on underrepresented classes (e.g., baseball, cheerleading).

---

## 📊 Dataset

- **Location:** `/content/dataset/`
- **Structure:**
  - `train/`: ~13,504 images
  - `valid/`: Estimated 10–20% of total dataset
  - `test/`: 500 images (5 per class × 100 classes)
- **Format:**
  - Images resized to 224×224
  - Labels from `sports.csv`
- **Challenges:** Some classes (e.g., baseball, baton twirling) are imbalanced.

---

## 🛠️ Methodology

### 1. Initial Model: MobileNetV2

- **Base:** Pre-trained on ImageNet
- **Custom Head:** GlobalAveragePooling + Dense layers
- **Training:**
  - Frozen base layers
  - Epochs: 10
  - Batch size: 32
  - Optimizer: Adam (lr=0.0001)
- **Augmentation:**
  - Rotation: 30°
  - Shift: 30%
  - Flip: Horizontal
  - Zoom: 20%

### 2. Improvements

- **Fine-Tuning MobileNetV2:**
  - Unfroze last 50 layers
  - Increased learning rate to 0.0005
  - Dropout reduced to 0.3
- **Optimization:**
  - Mixed precision (20–30% faster training)
  - Batch size: 16
  - Epochs: 30
  - Early Stopping (patience=5)
- **Augmentation Adjustments:**
  - Rotation: 15°
  - Shift: 10%
  - Zoom: 10%

### 3. EfficientNetB0 (Attempted)

- Pre-trained model `EfficientNetB0-100-(224 X 224)-98.40.h5`
- Error: `ValueError` due to `groups=1` (incompatible with TF 2.15+)

---

## 📈 Results

### Initial MobileNetV2

- **Train Accuracy:** 60.86%
- **Val Accuracy:** 84.20%
- **Test Accuracy:** 84.80%
- **Time:** ~130 minutes (GPU T4)
- **Weak F1-scores:**
  - baseball: 0.29
  - cheerleading: 0.33
  - baton twirling: 0.44

### Improved MobileNetV2

- **Train Accuracy:** 88.70%
- **Val Accuracy:** 93.60%
- **Test Accuracy:** **95.60%**
- **Time:** ~51 minutes (GPU T4)
- **F1-score Improvements:**
  - baseball: ↑ 0.75
  - cheerleading: ↑ 0.89

### Remaining Weak Classes

| Class              | F1-score |
|-------------------|----------|
| hydroplane racing | 0.75     |
| sky surfing       | 0.77     |

#### 📋 Sample Classification Report

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| air hockey       | 1.00      | 1.00   | 1.00     | 5       |
| baseball         | 1.00      | 0.60   | 0.75     | 5       |
| cheerleading     | 1.00      | 0.80   | 0.89     | 5       |
| ...              | ...       | ...    | ...      | ...     |
| **Accuracy**     |           |        | **0.96** | 500     |
| **Macro Avg**    | 0.96      | 0.96   | 0.96     | 500     |

---

## 📷 Visualizations

- ✅ Sample images from training/test sets
- ✅ Confusion matrix (100×100)
- ✅ Accuracy/loss plots across epochs

---

## 🚩 Challenges

- **Low Initial Train Accuracy:**
  - Caused by strong augmentation and frozen layers
  - ✅ Fixed with reduced augmentation + fine-tuning
- **Imbalanced Data:**
  - Some classes had poor F1-scores
  - ✅ Plan: Use class weights or oversampling
- **EfficientNetB0 Load Error:**
  - ❌ `groups` param error in `DepthwiseConv2D`
  - 🛠️ Solution: Downgrade TensorFlow or retrain model

---

## 🔜 Next Steps

- [ ] Fine-tune entire MobileNetV2
- [ ] Resolve EfficientNetB0 issue:
  ```bash
  pip install tensorflow==2.6.0
