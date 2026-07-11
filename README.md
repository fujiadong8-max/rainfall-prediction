# RainTomorrow: Australian Rainfall Prediction

A reproducible deep learning pipeline for binary rainfall prediction using the Australian **weatherAUS** dataset.

This project investigates the performance of different neural network architectures on structured meteorological data and implements a complete machine learning workflow, including data preprocessing, model training, evaluation, probabilistic inference, and downstream decision-making demonstrations.

Rather than serving as an isolated prediction model, this repository is designed as the weather prediction module of a larger intelligent transportation project. The predicted rainfall probabilities will be incorporated as external environmental states in a reinforcement learning framework for dynamic taxi pricing.

---

# Repository Highlights

- End-to-end PyTorch implementation
- Leakage-safe preprocessing pipeline
- Multiple neural network architectures
- Class imbalance handling using weighted loss
- Early stopping and reproducible experiments
- Batch probability inference
- Downstream decision-making demonstrations
- Modular code structure for future research

---

# Model Performance

Performance was evaluated on a held-out validation set using ROC AUC, Accuracy, Precision, Recall, and F1-score.

![Validation comparison](assets/model-comparison.png)

| Model | ROC AUC | Accuracy | Rain Precision | Rain Recall | Rain F1 |
|------|---------:|---------:|--------------:|-----------:|---------:|
| **MLP** | **0.900** | **0.811** | **0.552** | **0.822** | **0.660** |
| 1D CNN | 0.775 | 0.759 | 0.471 | 0.606 | 0.530 |

The multilayer perceptron (MLP) consistently outperformed the 1D CNN across all evaluation metrics. Since the weatherAUS dataset consists of tabular meteorological variables rather than sequential signals, fully connected architectures are better suited to this prediction task.

Because rainy days represent the minority class, overall accuracy alone is insufficient. ROC AUC, rain-class precision, recall, and F1-score provide a more informative assessment of model performance.

---

# Workflow

```text
weatherAUS.csv
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Imputation
        │
        ▼
Feature Encoding
        │
        ▼
Standardization
        │
        ▼
Train / Validation Split
        │
        ▼
PyTorch Models
(Logistic / MLP / Wide MLP / CNN)
        │
        ▼
Probability Prediction
        │
        ▼
Evaluation
        │
        ▼
Taxi Pricing Demo
Taxi Dispatch Demo
```

---

# Features

- Median imputation for numerical variables.
- Mode imputation for categorical variables.
- One-hot encoding of categorical features.
- Standardization of numerical variables.
- Stratified train/validation split.
- Leakage-safe preprocessing fitted only on training data.
- Multiple PyTorch model architectures.
- Weighted binary cross-entropy to mitigate class imbalance.
- Early stopping based on validation loss.
- Deterministic random seed for reproducibility.
- Automated model evaluation.
- Batch probability inference.
- Downstream decision simulations.

---

# Project Structure

```text
.
├── data.py                    # Data loading and preprocessing
├── model.py                   # Neural network architectures
├── train.py                   # Model training
├── evaluate.py                # Model evaluation
├── predict.py                 # Batch prediction
├── taxi_pricing_demo.py       # Pricing demonstration
├── taxi_dispatch_demo.py      # Dispatch demonstration
├── scripts/
│   └── make_readme_figures.py
├── assets/
│   ├── model-comparison.png
│   └── application-examples.png
└── outputs/
```

---

# Quick Start

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

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Download the dataset:

https://www.kaggle.com/datasets/mohamedmahmoud153/weatheraus

Place

```
weatherAUS.csv
```

under the project root.

---

# Train a Model

Train the recommended MLP model.

```bash
python train.py \
    --model mlp \
    --epochs 100 \
    --batch-size 256
```

Other supported architectures include:

- logistic
- mlp
- wide_mlp
- cnn

Example:

```bash
python train.py --model cnn
```

---

# Evaluate a Trained Model

```bash
python evaluate.py \
    --checkpoint outputs/mlp_model.pt \
    --csv-path weatherAUS.csv \
    --save-metrics outputs/mlp_eval.json
```

---

# Batch Prediction

```bash
python predict.py \
    --checkpoint outputs/mlp_model.pt \
    --csv-path weatherAUS.csv \
    --output outputs/mlp_predictions.csv
```

The generated CSV appends three new columns:

| Column | Description |
|---------|-------------|
| RainTomorrow_probability | Predicted probability of rain |
| RainTomorrow_pred | Binary prediction (0 or 1) |
| RainTomorrow_pred_label | Human-readable prediction ("Yes" / "No") |

---

# Downstream Decision Demonstrations

The repository contains two lightweight demonstrations illustrating how predicted rainfall probabilities can support operational decision making.

![Downstream examples](assets/application-examples.png)

## Dynamic Taxi Pricing

```bash
python taxi_pricing_demo.py \
    --predictions outputs/mlp_predictions.csv
```

This example adjusts fare multipliers according to predicted rainfall probability.

---

## Taxi Dispatch

```bash
python taxi_dispatch_demo.py \
    --predictions outputs/mlp_predictions.csv
```

This example simulates dispatch priority under different weather conditions.

These demonstrations are proof-of-concept examples intended to illustrate how probabilistic weather forecasts can be integrated into downstream intelligent transportation systems. They are **not** intended to estimate real-world business performance.

---

# Reproducing the Figures

Install matplotlib.

```bash
pip install matplotlib
```

Generate all README figures.

```bash
python scripts/make_readme_figures.py
```

---

# Technical Highlights

- PyTorch-based implementation
- Modular architecture for rapid experimentation
- Leakage-safe preprocessing
- Class imbalance mitigation
- Reproducible random seeds
- Early stopping
- Probability prediction
- JSON metric export
- CSV batch inference

---

# Discussion

## Current Limitations

Several methodological limitations should be considered.

- Random train-validation splitting may leak temporal or location-specific information.
- Threshold 0.5 is not optimized for deployment objectives.
- Probability calibration has not been evaluated.
- Results are reported using a single random seed.
- External validation has not been performed.
- Median and mode imputation cannot remove potential biases introduced by missing data.
- CNNs assume ordered feature structures, whereas tabular meteorological variables possess no natural spatial ordering.

Therefore, the CNN results should be interpreted as an architectural comparison rather than evidence that convolution is inherently suitable for this dataset.

---

# Data and Artifact Policy

The repository intentionally excludes:

- Raw weatherAUS dataset
- Trained model checkpoints
- Preprocessing artifacts
- Large prediction outputs

Only compact evaluation summaries, demonstration reports, and generated figures are retained to keep the repository lightweight while respecting the dataset's redistribution policy.

---

# Citation

If this repository contributes to your research or project, please cite it appropriately or reference the GitHub repository.

---

# License

No open-source license has currently been assigned.

The source code is provided for research demonstration and portfolio purposes. Reuse or redistribution requires permission from the repository owner. The weatherAUS dataset remains subject to the licensing terms of its original provider.
