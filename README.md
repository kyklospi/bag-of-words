# 📰 Sport News Similarity Analyzer

This Python script processes a collection of text files from BBC "sport" news dataset and finds the top 3 most similar documents to a given reference file using a Bag-of-Words (BoW) model and Euclidean distance.

## 🔍 What It Does

- Reads and loads all text files from the `bbc-news/sport` directory.
- Preprocesses text (lowercases, removes punctuation, splits into words).
- Builds a vocabulary of all unique words.
- Constructs BoW vectors for each document.
- Calculates Euclidean distance between the document `497.txt` and all others.
- Prints out the top 3 most similar documents along with a preview.

## 🧠 Key Functions

- `preprocess(text)`: Cleans and tokenizes the text.
- `create_bow(text)`: Converts a text into a BoW vector.
- `euclidean_distance(vec1, vec2)`: Calculates the Euclidean distance between two BoW vectors.

## ▶️ How to Run

Ensure you have your sport news dataset in `bbc-news/sport/`. Then run:

```bash
python bbc_news_sport_analysis.py
```

## 📌 Output

The script prints:
✅ First 1000 characters of the first document
✅ First 50 words in the sorted vocabulary
✅ Top 3 most similar documents to 497.txt, including:
- Document index
- Euclidean distance
- First 1000 characters of each similar document

## 🛠 Requirements

✅ Python 3.x
✅ No external libraries required (only uses built-in modules: os and math)