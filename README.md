# AI for Breast Cancer Diagnosis: Classification and Segmentation

This repository contains a series of projects applying advanced machine learning and deep learning techniques to breast cancer diagnosis. The notebooks progress from classical ML on tabular data to deep learning for image classification, and finally to advanced image segmentation.

This collection demonstrates a comprehensive workflow, including:

- ata Preprocessing: Cleaning and preparing tabular and image data.
- Model Building: Implementing, training, and fine-tuning a wide range of models.
- Advanced Architectures: Building ensembles, custom classifier heads, and Attention U-Nets.
- Robust Evaluation: Moving from a simple train/test split to 10-Fold Stratified Cross-Validation.
- Advanced Metrics: Using F1-score, Recall, Precision, IoU, and Dice Coefficient for nuanced evaluation.

## Projects & Notebooks

This repository is divided into three main themes:

1.  **Classical ML on Tabular Data**
2.  **Deep Learning Classification on Ultrasound Images**
3.  **Deep Learning Segmentation on Ultrasound Images**

### 1\. Classification (Tabular Data)

#### `01_ML_Classification_WDBC.ipynb`

  * **Task:** Classifies breast cancer (benign/malignant) using the classic Wisconsin Diagnostic Breast Cancer (WDBC) **tabular dataset**.
  * **Models:** `Logistic Regression`, `XGBoost`, `MLPClassifier`.
  * **Key Features:** A complete ML pipeline demonstrating data preprocessing, feature scaling, model comparison, in-depth feature importance analysis, and correlation heatmaps.

-----

### 2\. Classification (Ultrasound Images)

This is a series of notebooks, each building on the last to improve the classification of breast ultrasound images. These notebooks use a combination of the **BUSI** and **BrEaST** ultrasound datasets.

#### `02_Baseline_Classification_Ultrasound.ipynb`

  * **Task:** Establishes a baseline for classifying ultrasound images.
  * **Models:** Fine-tuned `ResNet50` and `Inception-v3` (using PyTorch/Timm).
  * **Key Features:** A complete pipeline from loading and splitting data, to training and evaluating deep learning models. Includes visualization of predictions and a confusion matrix.

#### `03_Ensemble_Classification_Ultrasound.ipynb`

  * **Task:** Improves on the baseline using an advanced **2-model weighted ensemble**.
  * **Models:** `EfficientNet-B0` + `ResNet34`.
  * **Key Features:** A custom training loop with weighted loss, gradient clipping, and other advanced regularization techniques to boost performance and prevent overfitting.

#### `04_Ensemble_Classification_Ultrasound_3_Models.ipynb`

  * **Task:** Extends the ensemble approach for maximum accuracy.
  * **Models:** A **3-model weighted ensemble** (`EfficientNet-B0` + `ResNet34` + `Inception-v3`).
  * **Key Features:** Demonstrates how to combine multiple diverse architectures to achieve a more robust and accurate final prediction.

#### `05_Cross_Validation_Classification_Ultrasound.ipynb`

  * **Task:** Implements a robust validation strategy, which is essential for any medical AI project.
  * **Model:** `ResNet50`.
  * **Key Features:** A complete **10-Fold Stratified Cross-Validation** pipeline. This provides a highly reliable measure of the model's true performance by training and testing on 10 different splits of the data and averaging the results.

-----

### 3\. Segmentation (Ultrasound Images)

#### `06_Segmentation_Attention_UNet.ipynb`

  * **Task:** Moves from *classification* (what is this?) to *segmentation* (where is it?). This model learns to draw a precise pixel-level mask around lesions.
  * **Model:** An **Attention U-Net with Inception Blocks** (using TensorFlow/Keras).
  * **Key Features:** A state-of-the-art architecture combining the U-Net's spatial preservation with Attention Gates (to focus on the tumor) and Inception Blocks (for multi-scale features).
  * **Metrics:** Implements advanced segmentation metrics (`IoU`, `Dice Coefficient`) and custom loss functions (`Dice Loss`, `Focal Tversky Loss`).

-----

## Datasets Used

1.  **Breast Ultrasound Images (BUSI & BrEaST):** The image-based notebooks (2-6) use a combination of the public [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and the `BrEaST-Lesions` dataset. This combined approach creates a larger, more diverse training set.
2.  **Wisconsin Diagnostic Breast Cancer (WDBC):** The tabular notebook (1) uses the [WDBC dataset from the UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

-----
