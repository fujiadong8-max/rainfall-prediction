# RainTomorrow: Australian Rainfall Prediction

A reproducible PyTorch pipeline for binary rainfall prediction using the Australian **weatherAUS** dataset.

This project implements an end-to-end deep learning workflow for predicting whether it will rain tomorrow based on historical meteorological observations. It covers data preprocessing, feature engineering, model training, evaluation, probability inference, and reproducible experimentation.

The repository is intended as a reproducible benchmark for tabular weather prediction using PyTorch and provides a modular codebase for future research and model development.

---

# Repository Highlights

- End-to-end PyTorch implementation
- Leakage-safe preprocessing pipeline
- Multiple neural network architectures
- Class imbalance handling using weighted loss
- Early stopping and reproducible experiments
- Batch probability inference
- Modular project structure
- Comprehensive evaluation metrics

---

# Model Performance

Performance was evaluated on a held-out validation set using ROC AUC, Accuracy, Precision, Recall, and F1-score.

![Validation comparison](assets/model-comparison.png)

| Model | ROC AUC | Accuracy | Rain Precision | Rain Recall | Rain F1 |
|------|---------:|---------:|--------------:|-----------:|---------:|
| **MLP** | **0.900** | **0.811** | **0.552** | **0.822** | **0.660** |
| 1D CNN | 0.775 | 0.759 | 0.471 | 0.606 | 0.530 |

Among the evaluated models, the multilayer perceptron (MLP) achieved the best overall performance. Since rainy days constitute the minority class, ROC AUC together with rain-class precision, recall, and F1-score provide a more informative assessment than accuracy alone.

---

# Workflow

```text
weatherAUS.csv
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Imputation
        │
        ▼
Categorical Encoding
        │
        ▼
Feature Standardization
        │
        ▼
Train / Validation Split
        │
        ▼
PyTorch Models
(Logistic / MLP / Wide MLP / CNN)
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
Probability Prediction
```

---

# Features

- Median imputation for numerical features.
- Mode imputation for categorical features.
- One-hot encoding for categorical variables.
- Standardization of numerical features.
- Leakage-safe preprocessing fitted only on the training set.
- Stratified train-validation split.
- Four PyTorch model architectures:
  - Logistic Regression
  - MLP
  - Wide MLP
  - 1D CNN
- Weighted binary cross-entropy for class imbalance.
- Early stopping based on validation loss.
- Deterministic random seeds for reproducibility.
- Batch inference on new datasets.
- Automatic evaluation and metric export.

---

# Project Structure

```text
.
├── data.py                    # Data loading and preprocessing
├── model.py                   # Neural network architectures
├── train.py                   # Model training
├── evaluate.py                # Model evaluation
├── predict.py                 # Batch inference
├── scripts/
│   └── make_readme_figures.py
├── assets/
│   └── model-comparison.png
├── outputs/
└── requirements.txt
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/fujiadong8-max/rainfall-prediction.git
cd rainfall-prediction
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Dataset

Download the Australian weather dataset from Kaggle:

https://www.kaggle.com/datasets/mohamedmahmoud153/weatheraus

Place

```
weatherAUS.csv
```

in the project root directory.

---

# Training

Train the recommended MLP model.

```bash
python train.py \
    --model mlp \
    --epochs 100 \
    --batch-size 256
```

Available models include:

- logistic
- mlp
- wide_mlp
- cnn

Example:

```bash
python train.py --model cnn
```

---

# Model Evaluation

Evaluate a trained model.

```bash
python evaluate.py \
    --checkpoint outputs/mlp_model.pt \
    --csv-path weatherAUS.csv \
    --save-metrics outputs/mlp_eval.json
```

Evaluation metrics include:

- ROC AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Batch Prediction

Generate probability predictions for an input dataset.

```bash
python predict.py \
    --checkpoint outputs/mlp_model.pt \
    --csv-path weatherAUS.csv \
    --output outputs/mlp_predictions.csv
```

The output file appends three new columns.

| Column | Description |
|---------|-------------|
| RainTomorrow_probability | Predicted probability of rainfall |
| RainTomorrow_pred | Binary prediction (0 or 1) |
| RainTomorrow_pred_label | Predicted class label ("Yes" or "No") |

---

# Reproducing the Figures

Install matplotlib.

```bash
pip install matplotlib
```

Generate all figures used in the README.

```bash
python scripts/make_readme_figures.py
```

---

# Technical Highlights

- Implemented entirely in PyTorch.
- Modular architecture for easy experimentation.
- Leakage-safe preprocessing.
- Class imbalance mitigation.
- Early stopping.
- Reproducible experiments.
- Batch probability inference.
- JSON metric export.
- Lightweight project organization.

---

# Discussion

The MLP achieved consistently better performance than the CNN across all evaluation metrics.

Although convolutional neural networks are highly effective for image and sequential data, transformed tabular meteorological features do not possess a natural spatial ordering. Consequently, fully connected architectures remain a more suitable choice for this prediction task.

The reported results demonstrate that a relatively simple MLP can achieve strong predictive performance when combined with appropriate preprocessing and regularization.

---

# Limitations

Several limitations should be considered.

- Random train-validation splitting may introduce temporal or location-specific information leakage.
- Only a single random seed is reported.
- Hyperparameter optimization is limited.
- Probability calibration has not been evaluated.
- Threshold 0.5 is not optimized for deployment objectives.
- External validation on independent weather datasets has not been performed.
- Median and mode imputation cannot eliminate potential bias caused by missing data.

Future work should address these limitations through more rigorous validation strategies.

---

# Data and Artifact Policy

To keep the repository lightweight and comply with the dataset's redistribution policy, the following files are excluded:

- Raw weatherAUS dataset
- Trained model checkpoints
- Preprocessing artifacts
- Large prediction outputs

Only lightweight evaluation summaries and generated figures are retained.

---

# Citation

If this repository contributes to your research, please consider citing or referencing the GitHub repository.

---

# License

No open-source license has currently been assigned.

The source code is provided for research, educational, and portfolio purposes. Redistribution or commercial reuse requires permission from the repository owner.

The weatherAUS dataset remains subject to the licensing terms of its original provider.
