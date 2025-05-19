# 🎬 IMDB Sentiment Analysis

This repository contains a sentiment analysis project using the IMDB movie review dataset. It involves preprocessing review text data, training a machine learning model (Logistic Regression or LSTM with Keras), and predicting whether a review is positive or negative.

---

## 📌 Overview

- **Dataset:** IMDB movie reviews (50,000 samples)
- **Goal:** Binary classification – predict if a review is positive (1) or negative (0)
- **Model Used:** Logistic Regression / LSTM (Keras)
- **Accuracy:** Achieved strong performance with both models

---

## 🧠 Features

- Data cleaning (removal of URLs, punctuation, and stopwords)
- Tokenization and sequence padding using Keras
- Model training and evaluation
- Save and load trained models
- Predict sentiment of new user input

---

## 📂 Project Structure

```

├── imdb\_sentiment\_analysis.ipynb   # Jupyter notebook with full analysis
├── data/
│   ├── train\_data.csv
│   └── test\_data.csv
├── models/
│   └── imdb\_model.h5 / imdb\_model.pkl
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the notebook or run your Python script:

```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```

---

## 🛠 Preprocessing Steps

* Convert to lowercase
* Remove URLs and special characters
* Remove stopwords using NLTK
* Tokenize text and pad sequences to equal length (200 words)

---

## 🧪 Model Options

### Option 1: Logistic Regression (Scikit-learn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Option 2: LSTM Model (TensorFlow/Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

---

## 💾 Model Saving & Loading

### Keras

```python
model.save('models/imdb_model.h5')
from tensorflow.keras.models import load_model
model = load_model('models/imdb_model.h5')
```

### Scikit-learn

```python
import joblib
joblib.dump(model, 'models/imdb_model.pkl')
model = joblib.load('models/imdb_model.pkl')
```

---

## 📈 Sample Prediction

```python
sample = ["This movie had brilliant acting and a powerful story."]
seq = tokenizer.texts_to_sequences(sample)
padded = pad_sequences(seq, maxlen=200)
prediction = model.predict(padded)
print("Positive" if prediction[0][0] > 0.5 else "Negative")
```

---

## 🧾 Requirements

Main libraries used:

* pandas
* numpy
* nltk
* tensorflow / keras
* scikit-learn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Jainil Desai**
Computer Science Student, Christ University
📧 [your.email@example.com](mailto:your.email@example.com)

---

## ✅ Status

✔️ Sentiment analysis model completed and evaluated
🔜 Optional: Add Streamlit or Flask interface for deployment

---

```

Let me know if you want to include the actual `requirements.txt`, Jupyter Notebook code, or convert it into a web app later using Streamlit or Flask!
```
