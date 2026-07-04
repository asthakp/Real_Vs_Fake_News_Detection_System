# Fake News Detection (NLP + ML)

An end-to-end system that classifies news articles as **real** or **fake** using sentence embeddings and an ensemble of classical ML models, deployed as an interactive Streamlit web app.

## Overview

Rather than relying on simple bag-of-words features, this project uses **sentence-transformer embeddings** to capture semantic meaning from article text, then trains and compares several classifiers on top of those embeddings — culminating in a voting ensemble that outperforms any single model.

## Dataset

- **Source:** [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data), Kaggle (Emine Bozkus, PhD) — `Fake.csv` / `True.csv`
- **Size after cleaning:** 44,689 articles (duplicates removed from both classes)
- **Labels:** `0` = Fake, `1` = Real

## Pipeline

### 1. Data Cleaning & EDA (`eda.ipynb`)
- Checked and removed null values and duplicate rows in both the fake and real datasets
- Merged the two datasets into a single labeled dataframe
- Verified class balance and inspected dataset shape
- Engineered features for exploration: title length, text length, word count
- Visualized text length distribution (fake vs. real) and subject-matter distribution across classes
- Cleaned text (whitespace/newline normalization) and combined `title` + `text` into a single field for modeling
- Exported the processed dataset for training

### 2. Feature Engineering & Modeling (`training.ipynb`)
- Encoded the cleaned title+text field into 384-dimensional embeddings using **`all-MiniLM-L6-v2`** (Sentence-Transformers)
- Split data 80/20 (stratified) into train/test sets
- Trained and compared five models on the embeddings:
  - Logistic Regression
  - Linear SVM
  - Random Forest
  - XGBoost
  - LightGBM
- Combined the three strongest models (Linear SVM, XGBoost, LightGBM) into a **hard-voting ensemble**
- Tuned the Linear SVM's hyperparameters (`C`, `tol`, `max_iter`) via `RandomizedSearchCV`
- Saved the final voting ensemble and the sentence-transformer embedder for reuse in the app

## Results

| Model | Accuracy | Fake F1 | Real F1 |
|---|---|---|---|
| Logistic Regression | 96.55% | 0.97 | 0.96 |
| Linear SVM | 97.53% | 0.98 | 0.97 |
| Random Forest | 89.08% | 0.90 | 0.88 |
| XGBoost | 97.53% | 0.98 | 0.97 |
| Tuned Linear SVM | 97.59% | 0.98 | 0.97 |
| **Voting Ensemble (SVM + XGBoost + LightGBM)** | **97.87%** | **0.98** | **0.98** |

The voting ensemble gave the best overall performance, edging out the tuned Linear SVM alone — combining models with different decision boundaries reduced the errors any single model made on its own. Random Forest lagged notably behind the other models, likely because it doesn't handle the dense, continuous embedding space as effectively as linear or boosting-based methods.

## Tech Stack

- **Language:** Python
- **NLP/Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Modeling:** scikit-learn (Logistic Regression, Linear SVM, Random Forest, Voting Classifier), XGBoost, LightGBM
- **Model persistence:** joblib
- **App/Deployment:** Streamlit

## Repository Structure

```
├── notebooks/
│   ├── eda.ipynb           # data cleaning & exploratory analysis
│   └── training.ipynb      # embedding, model training, tuning
├── app.py                  # Streamlit app (real-time prediction UI)
├── .env                     # environment variables (not committed with real values)
└── .gitignore
```

> Note: raw/processed data and saved models are excluded via `.gitignore` and generated locally by running the notebooks in order (`eda.ipynb` → `training.ipynb`).

## Getting Started

```bash
# Clone the repository
git clone https://github.com/asthakp/Real_Vs_Fake_News_Detection_System.git
cd Real_Vs_Fake_News_Detection_System

# Install dependencies
pip install pandas numpy matplotlib seaborn sentence-transformers scikit-learn xgboost lightgbm joblib streamlit

# Run the notebooks in order to generate the processed data and trained model
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/training.ipynb

# Launch the app
streamlit run app.py
```

## Future Improvements

- Experiment with fine-tuning a transformer model (e.g., DistilBERT) directly on the classification task rather than using frozen embeddings
- Add soft-voting or stacking to combine model probabilities instead of hard votes
- Expand the dataset with more recent news sources to reduce potential topic/time bias

## Author

**Astha Pandeya** — [GitHub](https://github.com/asthakp)

