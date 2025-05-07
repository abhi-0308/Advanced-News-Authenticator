# Advanced News Authenticator

**Advanced News Authenticator** is a Python-based machine learning project designed to classify news articles as real or fake. By applying natural language processing (NLP) techniques and machine learning algorithms, this project helps detect misinformation in news content.

## Features

* Classifies news articles as Fake or Real
* Implements machine learning models for prediction
* Preprocesses text data using TF-IDF vectorization
* Evaluates models using accuracy, confusion matrix, and classification report
* Interactive Jupyter Notebook for analysis and experimentation

## Models Used

* Logistic Regression
* Passive Aggressive Classifier
* (Optional) Naive Bayes
* (Optional) Support Vector Machine

## Preprocessing Steps

* Text cleaning and normalization
* Removal of stop words
* Tokenization of text
* TF-IDF vectorization for feature extraction

## Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Precision, Recall, and F1 Score

## Dataset

The project uses a dataset named `fake.csv` and `true.csv` which contains labeled news articles. Each entry includes the title, text, and a label indicating whether the news is real or fake.

## Requirements

* Python 3.x
* FLASK
* HTML/CSS
* JavaScript

## Future Improvements

* Develop a web-based interface for real-time classification
* Expand support for news in multiple languages
* Integrate live data using news APIs

