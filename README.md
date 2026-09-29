# IMDB Sentiment Analysis

Binary sentiment classification (positive / negative) of 50,000 IMDB movie reviews using TF-IDF features and Logistic Regression.

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) on Kaggle: 25,000 positive and 25,000 negative reviews.

The CSV (~60 MB) is not included in this repo. Download it from Kaggle and save it as `data/IMDB-Dataset.csv`.

## Project structure

```
imdb-sentiment-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb           # load, class balance, text cleaning, save cleaned CSV
│   └── 02_modeling_and_error_analysis.ipynb  # TF-IDF + LogReg, n-gram experiments, error analysis
└── images/
    ├── class_balance.png
    ├── ngram_comparison.png
    └── top_features.png
```

## How to run

```bash
pip install -r requirements.txt
```

Run the notebooks in order from the `notebooks/` folder. Notebook 01 writes `data/imdb_clean.csv`, which notebook 02 reads.
