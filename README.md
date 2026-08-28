# 🍎 Fruit & Vegetable Freshness Classifier

An AI-based image classification project that predicts whether a fruit or vegetable is **Fresh** or **Rotten**.

## Features

- Upload fruit/vegetable images
- CNN-based image classification
- Fresh/Rotten prediction
- Confidence percentage
- Streamlit web interface
- Model training and evaluation
- Confusion matrix and classification report

## Technologies

- Python
- TensorFlow / Keras
- CNN
- OpenCV
- Streamlit
- Scikit-learn

## Dataset Structure

Place your images inside:

```text
dataset/
├── train/
│   ├── fresh/
│   └── rotten/
├── validation/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
```

Use images that clearly represent fresh and rotten examples.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

After adding the dataset:

```bash
python train_model.py
```

The trained model will be saved as:

```text
model/freshness_model.keras
```

## Evaluate the Model

```bash
python evaluate_model.py
```

This displays accuracy, precision, recall, F1-score, and the confusion matrix.

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## GitHub

Initialize Git:

```bash
git init
git add .
git commit -m "Initial fruit freshness classifier"
```

Create a GitHub repository and connect it:

```bash
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git branch -M main
git push -u origin main
```

## Important

The repository contains the project structure and code, but **not a trained model or dataset**. Add your dataset and run `train_model.py` before using the classifier for predictions.
