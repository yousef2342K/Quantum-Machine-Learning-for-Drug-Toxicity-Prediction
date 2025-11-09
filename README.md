```markdown
# QuantumBoost2025: Advanced Quantum ML for Drug Toxicity Prediction

**Predicting drug toxicity using Quantum Variational Classifiers (QVC) and Quantum Support Vector Machines (QSVM)**  
Built with **Qiskit**, **scikit-learn**, and **imbalanced-learn**.

---

## Overview

This project implements a **state-of-the-art quantum machine learning pipeline** to predict **drug toxicity** (binary: `Non-Toxic=0`, `Toxic=1`) from molecular descriptors.

### Key Features
- **Advanced preprocessing**: missing value handling, variance filtering, correlation removal
- **Multi-stage feature selection**: Mutual Information + Random Forest importance
- **Class imbalance**: `BorderlineSMOTE` for robust oversampling
- **Quantum-ready scaling**: values mapped to `[0, 2π]`
- **Dimensionality reduction**: PCA → 6 qubits
- **Two quantum models**:
  - **QVC** (Variational Quantum Classifier)
  - **QSVM** (Quantum Support Vector Machine with fidelity kernel)
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1, MCC, Confusion Matrices
- **Modular 4-notebook pipeline**
- **Full reproducibility** via `requirements.txt`

---

## Project Structure

```
QuantumBoost2025/
│
├── Dataset/                    
│
├── notebooks/ 
├── 01_data_preprocessing.ipynb  # Load, clean, merge, save train_clean.csv
├── 02_feature_extraction.ipynb  # Correlation, MI, RF → X_selected.csv
├── 03_quantum_model_training.ipynb  # SMOTE, PCA, QVC/QSVM training
├── 04_evaluation_and_analysis.ipynb # Metrics, plots, comparison
│
├── requirements.txt             # Exact package versions
├── README.md                    
│
└──
```

---

## Requirements

- **Google Colab** (recommended) or local **Python 3.10–3.11**
- **Google Drive** access (for data and outputs)
- Internet connection (for `pip install`)

---

## Setup & Installation

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload all `.ipynb` files and `requirements.txt` to your Drive under:
   ```
   /MyDrive/QuantumBoost2025/
   ```
3. Place your dataset in:
   ```
   /MyDrive/QuantumBoost2025/Dataset/train_features.csv
   /MyDrive/QuantumBoost2025/Dataset/train_labels.csv
   ```
4. Open `01_data_preprocessing.ipynb` → **Run All**

> The first cell will install everything:
> ```python
> !pip install -r /content/drive/MyDrive/QuantumBoost2025/requirements.txt
> ```

### Option 2: Local Environment

```bash
# Clone or download the project
git clone <your-repo-url>
cd QuantumBoost2025

# Create and activate virtual environment
python -m venv quantum_env
source quantum_env/bin/activate  # Linux/Mac
# quantum_env\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Execution Order

Run the notebooks **in sequence**:

```bash
1. 01_data_preprocessing.ipynb
2. 02_feature_extraction.ipynb
3. 03_quantum_model_training.ipynb
4. 04_evaluation_and_analysis.ipynb
```

Each notebook:
- Loads inputs from previous stage
- Saves outputs to shared Drive folder
- Prints progress and final metrics

> **Tip**: Use `Runtime → Run all` in Colab for each notebook.

---

## Output Highlights

| File | Description |
|------|-------------|
| `model_comparison.csv` | Final QVC vs QSVM metrics |
| `model_comparison.png` | Bar chart of Accuracy, F1, MCC, etc. |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `qvc_model.pkl` / `qsvm_model.pkl` | Trained quantum models |
| `preprocessing_objects.pkl` | Full pipeline (scalers, PCA, features) |

---

## Making Predictions on New Data

Use the helper function (defined in original notebook or below):

```python
def predict_toxicity(model, new_data_df, prep_objects):
    X = new_data_df[prep_objects['features']]
    X = prep_objects['robust'].transform(X)
    X = prep_objects['minmax'].transform(X)
    X = prep_objects['pca'].transform(X)
    X = prep_objects['pca_scaler'].transform(X)
    return model.predict(X)
```

Example:
```python
with open('preprocessing_objects.pkl', 'rb') as f:
    prep = pickle.load(f)
with open('qvc_model.pkl', 'rb') as f:
    qvc = pickle.load(f)

new_molecules = pd.read_csv('test_features.csv')
preds = predict_toxicity(qvc, new_molecules, prep)
```

---

## Reproducibility

- All random seeds: `random_state=42`
- Exact library versions in `requirements.txt`
- Deterministic train/test split (`stratify=y`)
- No external APIs or mutable data sources

---

## Performance (Example)

| Model | Accuracy | Precision | Recall | F1 | MCC |
|-------|----------|-----------|--------|----|-----|
| QVC   | 0.842    | 0.801     | 0.756  | 0.778 | **0.678** |
| QSVM  | 0.835    | 0.788     | 0.742  | 0.764 | 0.659 |

> **Best model selected by Matthews Correlation Coefficient (MCC)**

---

## Contributors

- **Youssif Khalid Mahmoud Attia** – Quantum ML Engineer
- Built with [Qiskit](https://qiskit.org), [scikit-learn](https://scikit-learn.org), and [Google Colab](https://colab.research.google.com)

---

## License

MIT License – feel free to use, modify, and distribute.

---

**QuantumBoost2025** – *Where Quantum Meets Drug Discovery*
```