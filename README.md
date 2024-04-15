# Sentiment Analysis using Twitter Text Data

## Overview

This project implements sentiment analysis using Twitter text data to classify tweets as either positive or negative. It utilizes natural language processing (NLP) techniques and machine learning algorithms to analyze tweets and determine the sentiment expressed in each tweet.

## Key Components

### Data Preprocessing
- The project preprocesses the Twitter text data by removing symbols, converting text to lowercase, and Stemming words to their root form using the NLTK library.
- Stop words (commonly occurring words like "the", "is", "and", etc.) are removed to improve the quality of the text data.

### Feature Extraction
- Textual data is converted into numerical data using TF-IDF vectorization, which transforms text data into numerical vectors.
- The TF-IDF vectors represent the importance of each word in the tweet relative to the entire dataset.

### Model Training
- The project uses a logistic regression model to classify tweets into positive or negative sentiment categories.
- The model is trained on the preprocessed and vectorized Twitter text data.

### Model Evaluation
- The accuracy of the model is evaluated using both training and testing datasets.
- The accuracy score indicates the percentage of correctly classified tweets.

### Saving and Using the Model
- The trained model is saved using the pickle library for future use.
- Saved model files can be loaded and utilized to predict the sentiment of new tweets.

## How to Use

1. **Clone Repository:**
   Clone the project repository from GitHub to your local machine.

2. **Install Dependencies:**
   Ensure that Python and required libraries such as NumPy, pandas, scikit-learn, NLTK, etc., are installed on your system.

3. **Preprocess Data:**
   Preprocess the Twitter text data by removing symbols, converting text to lowercase, and stemming words using the provided script.

4. **Feature Extraction:**
   Convert preprocessed text data into numerical features using TF-IDF vectorization.

5. **Model Training:**
   Train a logistic regression model on the TF-IDF vectors to classify tweets into positive or negative sentiment categories.

6. **Model Evaluation:**
   Evaluate the accuracy of the trained model using both training and testing datasets.

7. **Save and Use Model:**
   Save the trained model using the pickle library for future predictions. Load the saved model and utilize it to predict the sentiment of new tweets.

## Contributors
- [Prathamesh Mahamuni](https://www.linkedin.com/in/prathameshmahamuni/) - Project Lead & Developer

The link to dataset : 
https://www.kaggle.com/datasets/kazanova/sentiment140
