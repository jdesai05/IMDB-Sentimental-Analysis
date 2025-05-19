# 🎬 IMDB Sentiment Analysis

This project uses machine learning to classify movie reviews from the IMDB dataset as positive or negative.

## 🗂 Project Structure

- `Model_Creation.ipynb` – Notebook to preprocess data, train, and save the model.
- `Model_Prediction.ipynb` – Load the trained model and make predictions.
- `IMDB Dataset.csv` – The dataset used for training and testing.
- `sentiment_model.pkl` – Saved sentiment analysis model.
- `vectorizer.pkl` – Saved TF-IDF vectorizer used during training.

## 🚀 How to Run

1. Clone the repository and install required packages:

```bash
pip install pandas scikit-learn nltk
````

2. Run the notebooks in order:

   * `Model_Creation.ipynb` to train and save the model.
   * `Model_Prediction.ipynb` to test predictions.

## ✅ Model

* **Vectorizer:** TF-IDF
* **Classifier:** Logistic Regression
* **Accuracy:** \~88%
