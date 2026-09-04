# 🎨 Banksy Street Art Authentication: A Multimodal Machine Learning Approach

> **A multimodal machine learning pipeline for authenticating Banksy street art, fusing a pre-trained ResNet-18 visual backbone with handcrafted physical attributes.**

## 📌 Project Overview
This project tackles a highly complex visual classification task in the built environment: authenticating Banksy's street art against noisy, severely weathered urban backgrounds. To achieve this, a **Dynamic Late-Fusion Multimodal Architecture** was developed, combining raw spatial semantics (pixels) with deterministic physical metadata (colorimetric variance, edge density, and spatial Shannon entropy).

## 🚀 Key Highlights & Critical Reflection
I recently completed this fascinating and highly challenging machine learning project. On the surface, it sounded like a standard binary classification task (Banksy vs. Non-Banksy). However, the real engineering challenge lay in ensuring absolute academic rigor.

Initially, our experimental results were deceptively perfect. The multimodal fusion network showed outstanding accuracy right out of the gate. But critical thinking prompted a deeper dive into the methodology. I soon realized that simple randomized tile-slicing had introduced a severe **"data leakage"** issue—patches from the exact same parent image were bleeding into both the training and validation sets. 

To resolve this, I tore down the pipeline and rebuilt it using a strict **"Parent-Image Level Isolation"** strategy and evaluated the model using an Image-Level voting mechanism on a perfectly balanced validation set.

### The Findings
The empirical results from this rigorous ablation study proved the validity of the fusion strategy under zero-leakage conditions:
* **Baseline 1 (Vision Only - ResNet-18):** 83.33% Accuracy
* **Baseline 2 (Attributes Only - 13-dim MLP):** 83.33% Accuracy
* **Ours (Dynamic Late Fusion):** **91.67% Accuracy**

*(Please see the confusion matrix and dynamic weight convergence plots below)*

![Confusion Matrix](Fig1_Confusion_Matrix.png)
*Figure 1: Image-level confusion matrices comparing the Vision Only baseline and our Robust Late Fusion model. The fusion network successfully corrects visual False Positives by incorporating statistical background data.*

![Weight Convergence](Fig2_Weight_Convergence.png)
*Figure 2: Convergence of the dynamic learnable weight during training, demonstrating the network's automatic balancing of visual and physical modalities.*

### Feature Space Collision
Through an in-depth error analysis, I discovered a phenomenon I call a **"feature space collision."** Extreme urban decay—such as dark water stains, peeling paint, and jagged plaster under dramatic lighting—produces high-contrast silhouettes and chaotic micro-edges. Mathematically, the spatial entropy and edge density of a ruined wall can mimic the quantitative profile of Banksy’s intentional stencil art. This project proved that differentiating intentional human artistic intervention from natural material degradation remains an incredibly complex challenge for automated defect recognition and heritage surveying.

## 🧠 Model Architecture
* **Visual Branch:** A pre-trained ResNet-18 backbone (ImageNet weights) with frozen shallow layers to prevent overfitting on specific wall textures.
* **Attribute Branch:** A 13-dimensional handcrafted statistical vector processed through an MLP (Linear -> BatchNorm -> Dropout -> ReLU).
* **Fusion Mechanism:** A dynamic, learnable residual weighting strategy optimized via a Sigmoid function: `Output = (1 - weight) * CNN_Logits + weight * MLP_Logits`.

## ⚙️ How to Run
The core pipeline is provided in the Jupyter Notebook (`.ipynb`). 
1. **Environment:** Python 3.8+, PyTorch, Torchvision, OpenCV, Scikit-learn, Pandas.
2. **Dataset:** Ensure the raw images are organized into `banksy/` and `not_banksy/` directories before running the slicing and isolation scripts.
3. **Execution:** The notebook is fully compatible with Google Colab. Enable the T4 GPU runtime for optimal training speed.

