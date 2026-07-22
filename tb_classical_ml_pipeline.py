# %% [markdown]
# # TB Chest X-ray Detection — Classical ML Pipeline
# Dataset: Kaggle "Tuberculosis (TB) Chest X-ray Database" (tawsifurrahman)
# Approach: HOG + GLCM feature extraction -> Decision Tree / Random Forest / SVM / KNN / Logistic Regression
# Class imbalance handled with SMOTE (on features) + class_weight='balanced' where supported

# %%
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from skimage.feature import hog, graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

from imblearn.over_sampling import SMOTE

# %% [markdown]
# ## 1. Configuration
# NOTE: Dataset lives in a SEPARATE folder (TB_Detection) from this code (TB_Classification),
# same pattern as the Brain Tumor project (mri/ folder vs Brain_Tumar_classification/ folder).
# This script itself should be saved inside: D:\AI project\TB_Classification\

# %%
# Dataset folder (raw data — kept separate from code)
DATASET_DIR = r"D:\AI project\TB_Detection\TB_Chest_Radiography_Database"
TB_FOLDER   = os.path.join(DATASET_DIR, "Tuberculosis")   # folder with TB images
NORMAL_FOLDER = os.path.join(DATASET_DIR, "Normal")       # folder with Normal images

# Output folder — everything this script generates (features, models, charts)
# will be saved HERE, inside the code folder, not inside the dataset folder
PROJECT_DIR = r"D:\AI project\TB_Classification"
os.makedirs(PROJECT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)   # resize target for consistent feature extraction
RANDOM_STATE = 42

# %% [markdown]
# ## 2. Feature Extraction Functions
# HOG captures shape/edge patterns, GLCM captures texture (useful for lung opacity/infiltration)

# %%
def extract_hog_features(img_gray):
    features, _ = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    return features

def extract_glcm_features(img_gray):
    # GLCM needs integer-valued image
    img_uint8 = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in props:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.array(feats)

def extract_combined_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0

    hog_feats = extract_hog_features(img)
    glcm_feats = extract_glcm_features(img)

    return np.concatenate([hog_feats, glcm_feats])

# %% [markdown]
# ## 3. Load Dataset & Extract Features

# %%
def load_dataset():
    X, y = [], []

    print("Extracting features from TB images...")
    for fname in tqdm(os.listdir(TB_FOLDER)):
        fpath = os.path.join(TB_FOLDER, fname)
        feats = extract_combined_features(fpath)
        if feats is not None:
            X.append(feats)
            y.append(1)  # TB = 1

    print("Extracting features from Normal images...")
    for fname in tqdm(os.listdir(NORMAL_FOLDER)):
        fpath = os.path.join(NORMAL_FOLDER, fname)
        feats = extract_combined_features(fpath)
        if feats is not None:
            X.append(feats)
            y.append(0)  # Normal = 0

    return np.array(X), np.array(y)

X, y = load_dataset()
print(f"\nTotal samples: {len(y)} | TB: {sum(y==1)} | Normal: {sum(y==0)}")
print(f"Feature vector length: {X.shape[1]}")

# Save raw features so you don't have to re-extract every time (extraction is slow)
np.save(os.path.join(PROJECT_DIR, "X_features.npy"), X)
np.save(os.path.join(PROJECT_DIR, "y_labels.npy"), y)

# %% [markdown]
# ## 4. Train-Test Split (stratified) + Scaling + SMOTE

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Before SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", np.bincount(y_train_bal))

# %% [markdown]
# ## 5. Define Models
# class_weight='balanced' added as extra safety net alongside SMOTE

# %%
models = {
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),  # KNN doesn't support class_weight, relies on SMOTE
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
}

# %% [markdown]
# ## 6. Train & Evaluate All Models

# %%
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)          # Most important — missing TB is dangerous
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })
    trained_models[name] = model

    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, f"cm_{name.replace(' ', '_')}.png"))
    plt.close()

# %% [markdown]
# ## 7. Compare All Models

# %%
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print(results_df.to_string(index=False))
print("\nNote: Sorted by F1-Score (balances Precision + Recall).")
print("KNN shows Recall=1.0 but very low Precision/Accuracy — it is over-predicting TB for almost everyone, not a reliable model.")

plt.figure(figsize=(10, 6))
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison — TB Chest X-ray Classification")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "model_comparison.png"))
plt.show()

# %% [markdown]
# ## 8. Save ALL Models (not just best) — app will show all 5 predictions side-by-side

# %%
best_model_name = results_df.iloc[0]["Model"]

# Save every trained model with a clean filename
model_filename_map = {
    "Decision Tree": "model_decision_tree.pkl",
    "Random Forest": "model_random_forest.pkl",
    "SVM (RBF)": "model_svm.pkl",
    "KNN": "model_knn.pkl",
    "Logistic Regression": "model_logistic_regression.pkl",
}

for name, model in trained_models.items():
    joblib.dump(model, os.path.join(PROJECT_DIR, model_filename_map[name]))

joblib.dump(scaler, os.path.join(PROJECT_DIR, "scaler.pkl"))

# Also save the comparison table so the app can display metrics without retraining
results_df.to_csv(os.path.join(PROJECT_DIR, "model_comparison.csv"), index=False)

print(f"\nAll 5 models saved in: {PROJECT_DIR}")
print(f"Best model (by Recall): {best_model_name}")
print("Scaler and comparison table also saved.")

# %% [markdown]
# ## 9. Inference Function (for new X-ray images / Streamlit app later)

# %%
def predict_tb(image_path, model_filename="model_svm.pkl"):
    """
    model_filename options: model_decision_tree.pkl, model_random_forest.pkl,
    model_svm.pkl, model_knn.pkl, model_logistic_regression.pkl
    """
    model = joblib.load(os.path.join(PROJECT_DIR, model_filename))
    scaler = joblib.load(os.path.join(PROJECT_DIR, "scaler.pkl"))

    feats = extract_combined_features(image_path)
    feats_scaled = scaler.transform(feats.reshape(1, -1))

    pred = model.predict(feats_scaled)[0]
    proba = model.predict_proba(feats_scaled)[0][1] if hasattr(model, "predict_proba") else None

    label = "Tuberculosis Detected" if pred == 1 else "Normal"
    print(f"Prediction: {label}" + (f" (confidence: {proba:.2%})" if proba is not None else ""))
    return label, proba

# Example usage:
# predict_tb("path/to/new_xray.png")