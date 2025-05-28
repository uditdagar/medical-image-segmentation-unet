# 🧠 Medical Image Segmentation using U-Net

This project performs semantic segmentation on **COVID-19 CT scan images** using the **U-Net architecture**, a popular deep learning model for biomedical image segmentation.

---

## 📌 Project Overview

- **Goal:** Accurately segment infected regions in lung CT scans of COVID-19 patients.
- **Architecture:** U-Net (convolutional encoder-decoder).
- **Loss Function:** Dice coefficient loss (to handle class imbalance).
- **Metrics:** Accuracy, Mean IoU, Dice Score.
- **Frameworks Used:** TensorFlow, Keras, OpenCV, NumPy.

---

## 📁 Dataset

- COVID-19 CT Scan dataset from Kaggle:  
  [COVID-19 CT Segmentation Dataset](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans)

- The dataset contains:
  - `frames/` — CT scan images.
  - `masks/` — Corresponding binary segmentation masks.

⚠️ **Note:** Dataset folders are excluded from this repository. You can download and place them in the root directory to train the model.

---

## 🚀 Model Training

- Images and masks are resized to `128x128`.
- U-Net model is compiled with:
  - `Adam` optimizer
  - `BinaryCrossentropy` loss
  - `Dice coefficient loss` added
  - `MeanIoU` for evaluation

- The best model is saved as `best_unet.h5`.

---

## ✅ Evaluation

After 20 epochs:
- **Accuracy:** ~0.994
- **Loss:** ~0.012
- **Mean IoU:** ~0.49
- **Dice Score:** ~0.77


