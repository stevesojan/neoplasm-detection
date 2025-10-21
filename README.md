
# Tumor Classification Using an Ensemble of ShuffleNet_V2, MobileNet_V2, and ResNet18

---

##  Abstract

Breast cancer remains one of the leading causes of mortality among women worldwide.  
This project presents a **lightweight ensemble deep learning model** that classifies tumors as *benign* or *malignant* from mammographic images.  

Three efficient convolutional neural networks—**ShuffleNet_V2**, **MobileNet_V2**, and **ResNet18**—are trained independently and combined using a **bagging ensemble** technique.  
The ensemble averages their softmax outputs to form robust predictions.  

To improve interpretability, the pipeline also visualizes **side-by-side Craniocaudal (CC)** and **Mediolateral Oblique (MLO)** projections during validation.  

---

##  1. Introduction

Breast cancer screening typically uses two standard mammogram views—CC and MLO.  
Given the subtle differences in tissue patterns, automated classification can be difficult.  
By combining the **strengths of multiple CNNs**, this project enhances prediction accuracy while maintaining computational efficiency.

---

##  2. Dataset

The dataset contains mammographic images labeled as **benign** or **malignant**.  
Each sample includes both **CC** and **MLO** images, offering complementary perspectives of breast tissue.  

---

## 🧩 3. Model Architecture Overview

### 3.1 ShuffleNet_V2
- **Key Features:**
  - *Channel Shuffle* for inter-group feature communication.  
  - *Grouped Convolutions* to reduce computational cost.  
  - *Lightweight design* suited for deployment on edge devices.  
- **Contribution to Ensemble:** Extracts high-level features efficiently.  

![ShuffleNet_V2 architecture image here from](/images/shufflenet_v2.png)

---

### 3.2 MobileNet_V2
- **Key Features:**
  - *Depthwise Separable Convolutions* reduce parameters dramatically.  
  - *Inverted Residuals* with linear bottlenecks for compact yet expressive modeling.  
- **Contribution to Ensemble:** Balances efficiency and accuracy with mid-level feature extraction.  

![MobileNet_V2 architecture image here from](/images/mobilenet_v2.png)

---

### 3.3 ResNet18
- **Key Features:**
  - *Skip Connections* to avoid vanishing gradients.  
  - *17 Convolutional Layers + 1 FC layer* for deeper representation learning.
  - *The 18 in ResNet-18 refers to the no. of layers that have learnable parameters i.e 17 conv layers and 1 FC layer*  
- **Contribution to Ensemble:** Captures complex hierarchical features, complementing lighter models.  

![ResNet18 architecture image here from](/images/resnet18.png)

---

## ⚙️ 4. Ensemble Model Formation

Each model is trained individually. During inference, the ensemble prediction is computed as follows:

1. Each model outputs a probability distribution over the classes using **softmax**.  
2. The outputs are averaged element-wise:  

   $$
   P_{\text{final}} = \frac{1}{3} \sum_{i=1}^{3} P_i
   $$

3. The predicted class is chosen as:

   $$
   \hat{y} = \arg\max (P_{\text{final}})
   $$

This represents a **bagging ensemble** strategy, reducing variance and improving generalization.

---

##  5. Implementation

The project was implemented using **Python 3.12.8** and **PyTorch 2.6.0** on **PyCharm Professional**.

### Key Steps:
1. **Dataset Preparation:**  
   Each mammogram sample contains both CC and MLO views, preprocessed with resizing, augmentation, and normalization.

2. **Training Setup:**  
   - Epochs: 10  
   - Optimizer: Adam (learning rate = 1e-4)  
   - Loss: Cross-Entropy  
   - Batch Size: 16  
   - Device: CUDA or CPU automatically detected  

  ![Ensemble Accuracy per epoch image here from](/images/ensemble_acc.png)

3. **Visualization:**  
   - Displays side-by-side CC and MLO images with predictions and ground truths.  
   - Generates a **confusion matrix** and **classification report** post-validation.

  ![Prediction O/P image here from](/images/prediction_op.png)
---

##  6. Evaluation Metrics

### 6.1 Confusion Matrix

| Actual / Predicted | Benign | Malignant |
|--------------------|---------|------------|
| **Benign**         | 15       | 5          |
| **Malignant**      | 5       | 25          |


### 6.2 Classification Report

| Class      | Precision | Recall | F1-score | Support |
|-------------|-----------|---------|-----------|----------|
| Benign      | 0.75      | 0.75    | 0.75      | 20        |
| Malignant   | 0.83      | 0.83    | 0.83      | 30        |
| **Accuracy**| **0.80**  |         |           | 50       |
| Macro avg   | 0.79      | 0.79    | 0.79      | 50       |
| Weighted avg| 0.80      | 0.80    | 0.80      | 50       |

### Metric Definitions

- **Precision:**  
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **Recall:**  
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

- **F1-Score:**  
  $$
  F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

---

##  7. Conclusion

The ensemble of **ShuffleNet_V2**, **MobileNet_V2**, and **ResNet18** effectively enhances tumor classification accuracy from mammograms.  
This hybrid approach integrates:
- The efficiency of ShuffleNet,  
- The balance of MobileNet, and  
- The depth of ResNet.  

Visual interpretability using CC and MLO projections and robust evaluation metrics confirm the **clinical applicability** of the approach.  
Future work can explore **data augmentation** and **transfer learning** with larger datasets for even better generalization.

---


