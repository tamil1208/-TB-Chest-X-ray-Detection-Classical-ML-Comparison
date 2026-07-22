# 🫁 TB Chest X-ray Detection — Classical ML Comparison

A machine learning web app that screens chest X-rays for **Tuberculosis (TB)** using classical ML algorithms — not deep learning. Instead of a single black-box model, this project trains **5 different algorithms** and runs **4 of them live** in the app, side-by-side, for transparency and comparison.

🔗 **Live Demo:** https://tb-chest-xray-detection-classical-ml.onrender.com/

---

## 📊 Overview

- **Dataset:** [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) (Kaggle) — 3,500 Normal + 700 TB chest X-ray images
- **Approach:** Classical Machine Learning (no CNN/deep learning)
- **Features used:**
  - **HOG** (Histogram of Oriented Gradients) — shape and edge patterns
  - **GLCM** (Gray Level Co-occurrence Matrix) — texture patterns
- **Class imbalance handling:** SMOTE oversampling + `class_weight='balanced'`

---

## 🤖 Models Trained & Compared

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (RBF)** | 98.8% | 98.5% | 94.3% | 96.4% | 0.998 |
| **Logistic Regression** | 98.7% | 96.4% | 95.7% | 96.1% | 0.997 |
| **Random Forest** | 97.0% | 94.6% | 87.1% | 90.7% | 0.991 |
| **Decision Tree** | 87.7% | 61.8% | 69.3% | 65.3% | 0.804 |
| **KNN** | 59.6% | 29.2% | 100.0% | 45.2% | 0.899 |

> ⚠️ **Note on KNN:** Despite 100% recall, KNN has very low precision on this dataset — it over-predicts TB for almost every image. It was trained and evaluated like the others, but is **not included in live predictions** in the deployed app: its saved model file is ~347MB (KNN stores its entire training set), too large to deploy. Its results are still shown above for transparency.

---

## 🖥️ App Features

- Upload any chest X-ray (PNG/JPG)
- See predictions from **4 live models simultaneously** (SVM, Logistic Regression, Random Forest, Decision Tree)
- Majority-vote summary ("X out of 4 models predict TB")
- Per-model confidence percentage
- Visual bar chart comparing all live models
- Sidebar with full performance metrics table for all 5 trained models (including KNN)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Feature Extraction:** scikit-image (HOG, GLCM)
- **ML Models:** scikit-learn (Decision Tree, Random Forest, SVM, KNN, Logistic Regression)
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Web App:** Streamlit
- **Visualization:** Plotly
- **Deployment:** Render

---

## 📂 Project Structure

```
├── app.py                          # Streamlit web app
├── tb_classical_ml_pipeline.py     # Full training pipeline (feature extraction + training for all 5 models)
├── requirements.txt                # Python dependencies
├── model_svm.pkl                   # Trained SVM model
├── model_logistic_regression.pkl   # Trained Logistic Regression model
├── model_random_forest.pkl         # Trained Random Forest model
├── model_decision_tree.pkl         # Trained Decision Tree model
├── scaler.pkl                      # StandardScaler used for feature normalization
└── model_comparison.csv            # Saved performance metrics for all 5 trained models
```

> Note: `model_knn.pkl` is generated locally by the training pipeline but is not included in this repo (~347MB, exceeds GitHub's file size limits).

---

## ▶️ Run Locally

```bash
# Clone the repo
git clone https://github.com/tamil1208/-TB-Chest-X-ray-Detection-Classical-ML-Comparison-ML.git
cd TB-Chest-Xray-Detection-Classical-ML

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**. It is **not a certified diagnostic tool** and should never be used as a substitute for professional medical evaluation. Always consult a radiologist or physician for actual TB diagnosis.

---

## 👩‍💻 Author

**Tamilarasan**
Data Science & AI Portfolio 

