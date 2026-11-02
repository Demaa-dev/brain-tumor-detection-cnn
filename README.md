Brain Tumor Detection from MRI Scans using Deep Learning
**Deep Learning Project**

Project Overview
This project develops a deep learning system to automatically detect brain tumors from MRI scans. After establishing a baseline Custom CNN, we implemented Transfer Learning (MobileNetV2) to overcome the limitations of a small dataset (253 images). The model classifies MRI images into two categories:

Tumor – Brain tumor present
No Tumor – No brain tumor detected
A web application built with Streamlit allows users to upload MRI images and receive instant predictions.



## Repository Structure
brain-tumor-detection-cnn/
│
├── app.py ← Streamlit web application
├── brain_tumor_model_mobilenet.keras ← Trained MobileNetV2 model (Best Performer)
├── requirements.txt ← Python dependencies
├── README.md ← This file
└── BrainTumorDetection_Report.docx ← Full project report & analysis

*(Note: The training pipeline `brain_tumor_cnn.py` is executed in Google Colab to generate the models and graphs found in the report).*

## Setup & Installation

### Run Web App Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-detection-cnn.git
cd brain-tumor-detection-cnn

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py

---

## Dataset

| Attribute | Details |
|---|---|
| Name | Brain MRI Images for Brain Tumor Detection |
| Source | [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |
| Total Samples | 253 MRI images |
| Classes | Tumor (155), No Tumor (98) |
| Image Size | Resized to 224 × 224 pixels |
| Split | 60% Train (151) / 20% Validation (51) / 20% Test (51) |



---

## Final Model Architecture (MobileNetV2 + Transfer Learning)
To prevent overfitting on a small dataset, we froze pre-trained ImageNet weights and added a custom classification head:

Base Model: MobileNetV2 (Frozen)
Classifier Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
Class Weights: Applied balanced weights to handle class imbalance (155 vs 98).

## Results
The project conducted three distinct experiments to find the most honest and robust architecture:
| Metric | Custom CNN (Baseline) | Custom CNN (Weighted) | **MobileNetV2 (Weighted)** |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 68.63% | 64.71% | **80.39%** |
| **Precision** | 67.44% | 68.57% | **80.00%** |
| **Recall** | 93.55% | 77.42% | **90.32%** |
| **F1-Score** | 78.38% | 72.73% | **84.85%** |


Note: The baseline Custom CNN suffered from a high false-positive rate (14 healthy brains flagged as tumors) due to class imbalance. Applying class weights to the small custom model caused loss divergence. MobileNetV2 successfully balanced precision and recall, cutting the false-positive rate in half.
---


## Disclaimer

This application is a **research prototype** for educational purposes only. It is **NOT** a certified medical device and should **NOT** replace professional medical diagnosis.

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
