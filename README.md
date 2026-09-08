# Hybrid Model for Skin Disease Classification with Explainable AI

## 📋 Overview

This project implements a comprehensive skin disease classification system using deep learning models and explainable AI techniques. The system classifies skin lesions into 7 categories using the HAM10000 dataset and provides visual explanations for model predictions through Grad-CAM, LIME, and SHAP techniques.

### 🎯 Project Goals
- **Classification**: Accurately classify skin lesions into 7 categories
- **Model Comparison**: Compare performance of EfficientNet, Vision Transformer, and hybrid approaches
- **Explainability**: Provide interpretable results using multiple explainability techniques
- **Accessibility**: Make AI predictions understandable to medical professionals

---

## 🏥 Dataset

### HAM10000 - Skin Cancer MNIST
**Dataset Structure:**
- **Total Samples**: 10,015 dermatoscopic images
- **Image Resolution**: Varies (resized to 72x72 or 224x224 depending on model)
- **Classes**: 7 skin lesion types

### Disease Categories
| Code | Disease | Description |
|------|---------|-------------|
| mel | Melanoma | Most dangerous skin cancer |
| nv | Nevus | Common mole |
| bkl | Keratosis | Non-cancerous skin growth |
| bcc | Basal Cell Carcinoma | Common skin cancer |
| akiec | Actinic Keratosis | Precancerous lesion |
| vasc | Vascular Lesion | Blood vessel disorder |
| df | Dermatofibroma | Benign skin tumor |

**Data Location**: `archive/HAM10000_metadata.csv` and images split across `HAM10000_images_part_1/` and `HAM10000_images_part_2/`

---

## 🧠 Models

### 1. **EfficientNetB0** (`EfficientNetB0.py`)
**Architecture**: Efficient Convolutional Neural Network
- **Framework**: TensorFlow/Keras
- **Input Size**: 72×72 pixels
- **Base Model**: EfficientNetV2B0 with ImageNet pretrained weights
- **Fine-tuning**: Full model fine-tuning enabled
- **Learning Rate**: 0.0001
- **Epochs**: 12 (with early stopping)
- **Batch Size**: 32

**Key Features:**
- Fast inference time
- Low memory footprint
- Effective feature extraction through compound scaling
- Pre-trained on ImageNet

**Output Files**:
- `efficientnet_model.h5` - Keras model format
- `efficientnet_model.pth` - PyTorch model format
- `efficientnet_model.onnx` - ONNX format for cross-platform deployment

---

### 2. **Vision Transformer (ViT)** (`VisionTransformer.py`)
**Architecture**: Pure Transformer-based approach
- **Framework**: PyTorch
- **Input Size**: 72×72 pixels
- **Patch Size**: 6×6 (144 patches total)
- **Embedding Dimension**: 64
- **Attention Heads**: 4
- **Transformer Layers**: 8
- **Learning Rate**: 1e-4
- **Epochs**: 20
- **Batch Size**: 32

**Key Components:**
- **PatchEmbedding**: Converts image patches to embeddings
- **MultiHeadSelfAttention**: Self-attention mechanism with 4 heads
- **MLP Block**: Feed-forward network with hidden dimension = 2× embed_dim
- **Positional Encoding**: Learnable positional embeddings
- **Classification Head**: Linear layer mapping to 7 classes

**Advantages:**
- Global receptive field from the start
- Better generalization on various image resolutions
- State-of-the-art performance on vision tasks

---

### 3. **Hybrid Model** (`CombinedModel.py` / `new-hybrid-model.py`)
**Architecture**: Ensemble combining EfficientNet and Vision Transformer
- **Framework**: PyTorch
- **Input Size**: 224×224 pixels
- **Batch Size**: 32

**Model Combination Strategy:**
- Both EfficientNet and ViT process images independently
- Features are extracted from pre-trained models
- Predictions are combined via voting or weighted ensemble
- Enhanced robustness through multi-model consensus

**Benefits:**
- Leverages strengths of both CNN and Transformer architectures
- Improved accuracy through ensemble voting
- Better generalization and robustness

---

## 🎨 Explainable AI Techniques

### 1. **Grad-CAM** (`gradcam_explainer.py`, `explainers.py`)
**Gradient-weighted Class Activation Mapping**

**How It Works:**
- Visualizes which regions of the image influenced the model's prediction
- Uses gradients flowing into the last convolutional layer
- Generates heatmaps highlighting important features

**Output**:
- Heatmap overlay on original image
- Circle annotations for peak activation regions
- Saved to `efficientNet_explainers_outputs/`

**Use Case**: Quick understanding of what features the model focuses on

---

### 2. **LIME** (`lime_explainer.py`, `explainers.py`)
**Local Interpretable Model-agnostic Explanations**

**How It Works:**
- Treats model as "black box" - model-agnostic approach
- Creates perturbed versions of the input image
- Uses superpixels to segment regions of interest
- Highlights which superpixels contributed positively/negatively

**Output**:
- Yellow filled regions for defective/important areas
- Boundary-marked superpixel visualization
- Local linear model approximation
- Saved to `efficientNet_explainers_outputs/`

**Use Case**: Understanding which image regions support the prediction
**Note**: Takes 30-60 seconds per image

---

### 3. **SHAP** (`shap_explainer.py`, `explainers.py`)
**SHapley Additive exPlanations**

**How It Works:**
- Game theory-based approach using Shapley values
- Calculates feature importance by considering all feature combinations
- Superpixel-based implementation for image data
- Provides consistent and theoretically sound explanations

**Output**:
- Superpixel importance scores
- Visual explanations with contribution magnitudes
- Saved to `efficientNet_explainers_outputs/`

**Use Case**: Rigorous, theoretically-grounded explanation of model decisions
**Note**: Computationally intensive

---

## 📁 File Structure

```
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── check_data.py                      # Data validation utility
├── utils.py                           # Shared utilities (model/image loading)
│
├── Models/
│   ├── EfficientNetB0.py             # EfficientNet implementation (TensorFlow)
│   ├── VisionTransformer.py          # ViT implementation (PyTorch)
│   ├── CombinedModel.py              # Hybrid model (early version)
│   └── new-hybrid-model.py           # Hybrid model (updated)
│
├── Explainers/
│   ├── explainers.py                 # Unified explainer module (all 3 techniques)
│   ├── gradcam_explainer.py          # Grad-CAM standalone
│   ├── lime_explainer.py             # LIME standalone
│   └── shap_explainer.py             # SHAP standalone
│
├── Pretrained Models/
│   ├── efficientnet_model.h5         # Keras format
│   ├── efficientnet_model.pth        # PyTorch format
│   └── efficientnet_model.onnx       # ONNX format
│
├── Models pics/
│   ├── Efficient_Net/                # Architecture diagrams
│   ├── VisionTransformer/            # Architecture diagrams
│   ├── Hybrid_Model/                 # Architecture diagrams
│   └── Explainable_AI/               # Visualization examples
│
├── efficientNet_explainers_outputs/  # Explanation outputs
└── visionTransformer_explainers_outputs/  # Explanation outputs
```

---

## 📦 Dependencies

**Core Libraries:**
- `torch` - PyTorch framework
- `tensorflow` - TensorFlow/Keras framework
- `torchvision` - PyTorch vision utilities
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` & `seaborn` - Visualization
- `Pillow` - Image processing
- `scikit-learn` - ML utilities and metrics

**Explainability Libraries:**
- `lime` - Local interpretable explanations
- `shap` - SHAP value calculations
- `scikit-image` - Image segmentation for LIME/SHAP

**Misc:**
- `imageio` - Image I/O
- `opencv-python-headless` - Computer vision operations
- `torchmetrics` - Metric calculations

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository and navigate to project
cd code

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset
```bash
# Check if all data files are available
python check_data.py
```

### 3. Train Models

**EfficientNetB0 (TensorFlow)**
```bash
python EfficientNetB0.py
```

**Vision Transformer (PyTorch)**
```bash
python VisionTransformer.py
```

**Hybrid Model (PyTorch)**
```bash
python new-hybrid-model.py
```

### 4. Generate Explanations

**Using Grad-CAM**
```python
from gradcam_explainer import GradCAMExplainer
explainer = GradCAMExplainer()
explainer.run_on_image('path/to/image.jpg')
```

**Using LIME**
```python
from lime_explainer import LIMEExplainerWrapper
explainer = LIMEExplainerWrapper()
explainer.explain('path/to/image.jpg')
```

**Using SHAP**
```python
from shap_explainer import SHAPExplainer
explainer = SHAPExplainer()
explainer.explain('path/to/image.jpg')
```

---

## 🔧 Configuration

### Dataset Paths
Update the following paths in each script to match your local setup:
```python
DATA_DIR = r"path/to/HAM10000_metadata.csv"
img_dir_base = r"path/to/archive"
```

### Model Hyperparameters

**EfficientNetB0:**
- `batch_size`: 32
- `num_epochs`: 12
- `learning_rate`: 0.0001
- `image_size`: 72×72

**Vision Transformer:**
- `batch_size`: 32
- `num_epochs`: 20
- `learning_rate`: 1e-4
- `image_size`: 72×72
- `patch_size`: 6
- `embed_dim`: 64
- `num_heads`: 4
- `num_layers`: 8

**Hybrid Model:**
- `batch_size`: 32
- `image_size`: 224×224
- Both component models fine-tuned

---

## 📊 Model Comparison

| Metric | EfficientNet | Vision Transformer | Hybrid |
|--------|--------------|-------------------|--------|
| **Framework** | TensorFlow | PyTorch | PyTorch |
| **Input Size** | 72×72 | 72×72 | 224×224 |
| **Training Speed** | Fast | Slow-Medium | Medium |
| **Inference Speed** | Very Fast | Fast | Medium |
| **Memory Usage** | Low | Medium | High |
| **Architecture Type** | CNN | Transformer | Ensemble |
| **Global Context** | Limited | Excellent | Excellent |
| **Expected Accuracy** | High | Very High | Highest |

---

## 🎓 Training Pipeline

### Data Preprocessing
1. **Loading**: Images loaded from two directories (part 1 & part 2)
2. **Resizing**: 72×72 for EfficientNet/ViT, 224×224 for hybrid
3. **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Data Augmentation**: Random resizing, cropping (model-specific)

### Train/Validation Split
- **Training**: 80% of data
- **Validation**: 20% of data
- **Stratification**: Maintains class distribution

### Loss & Optimization
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate Scheduling**: Optional early stopping based on validation loss

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance visualization
- **Training Curves**: Loss and accuracy over epochs

---

## 📈 Outputs

### Model Outputs
- Trained model weights (`.h5`, `.pth`, `.onnx`)
- Training history plots
- Confusion matrix visualizations

### Explanation Outputs (saved in subdirectories)
- **Grad-CAM**: Heatmap overlays with peak activation circles
- **LIME**: Superpixel-based explanations with marked boundaries
- **SHAP**: Feature importance scores and visualizations

---

## 💡 Key Insights

### EfficientNetB0
- ✅ Fastest training and inference
- ✅ Excellent accuracy with minimal computational cost
- ❌ Limited global context (CNN limitations)
- **Best for**: Resource-constrained environments, real-time applications

### Vision Transformer
- ✅ Superior accuracy on diverse inputs
- ✅ Global receptive field from start
- ❌ Slower training/inference
- ❌ Requires more data for optimal performance
- **Best for**: High-accuracy requirements, good data availability

### Hybrid Model
- ✅ Combines strengths of both architectures
- ✅ Highest expected accuracy
- ✅ Robust predictions via ensemble voting
- ❌ Highest computational cost
- ❌ Complex to maintain
- **Best for**: High-stakes medical applications, maximum accuracy

---

## 🔍 Explainability Comparison

### Grad-CAM
- **Speed**: Fastest (~1 second)
- **Interpretability**: Good visual heatmaps
- **Limitation**: Only shows "where" the model looked
- **Best for**: Quick diagnostics, model debugging

### LIME
- **Speed**: Moderate (~30-60 seconds)
- **Interpretability**: Intuitive superpixel explanations
- **Advantage**: Model-agnostic (works with any model)
- **Best for**: Stakeholder communication, local understanding

### SHAP
- **Speed**: Slowest (~1-2 minutes)
- **Interpretability**: Theoretically sound Shapley values
- **Advantage**: Consistent, game-theory based
- **Best for**: Research, rigorous scientific analysis

---

## ⚠️ Important Notes

### Dataset Paths
The code uses absolute paths for Windows systems. Update these before running:
```python
# Windows paths (use raw strings with r prefix)
DATA_DIR = r"E:\path\to\HAM10000_metadata.csv"
img_dir_base = r"E:\path\to\archive"
```

### Pretrained Weights
The hybrid model expects cached PyTorch weights:
```python
vit_weights_path = r"C:\Users\YourUsername\.cache\torch\hub\checkpoints\vit_b_16-c867db91.pth"
eff_weights_path = r"C:\Users\YourUsername\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth"
```

### GPU/CPU Usage
All models automatically detect CUDA availability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Explanation Processing
- Explainers are computationally intensive
- LIME and SHAP are recommended for individual image analysis
- Batch processing requires careful memory management

---

## 📚 References

### Deep Learning Models
- **EfficientNet**: Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for CNNs"
- **Vision Transformer**: Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words"

### Explainability Techniques
- **Grad-CAM**: Selvaraju et al. (2016) - "Grad-CAM: Visual Explanations from Deep Networks"
- **LIME**: Ribeiro et al. (2016) - "Why Should I Trust You?" Local Explanations of Black Box Models"
- **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

### Dataset
- **HAM10000**: Tschandl et al. (2018) - "The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images"

---

## 📝 License
See LICENSE file for details

---

## 👥 Project Contributors
- **Master's Program**: AI - Sem 1
- **Project Type**: Skin Disease Classification with Explainable AI

---

## 🤝 Support & Issues

For issues or questions:
1. Check if dataset paths are correctly configured
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure image data exists in specified directories
4. Check GPU availability: `torch.cuda.is_available()` / `torch.cuda.get_device_name()`

---

**Last Updated**: December 2025
**Repository**: Hybrid-Model-For-Skin-Disease-Classification-With-Explainable-AI
