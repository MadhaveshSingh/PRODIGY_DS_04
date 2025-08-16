# PRODIGY_DS_04

Sentiment Analysis on Social Media Data

This project provides a comprehensive pipeline for analyzing and visualizing sentiment patterns in social media datasets. The script (sentiment_analysis_social_media.py) is designed to automatically detect text and sentiment label columns, clean raw text, explore key patterns, and train a simple machine learning model to classify sentiments.

**Features :-**

**1. Automatic column detection :-** Identifies likely text and sentiment columns (with fallback heuristics).<br>  
**2. Text preprocessing :-** Cleans tweets/posts by removing URLs, mentions, hashtags, punctuation, and extra spaces.<br>  
**3. Exploratory analysis :-**  
<br>Sentiment distribution visualization.  
Time trend analysis (if timestamp columns exist).  
Hashtag frequency and top hashtags per sentiment.  
N-gram analysis (unigrams and bigrams).

**4. Machine learning model :-**

TF-IDF vectorization + Logistic Regression.  
Evaluation using classification report & confusion matrix.

**5. Output generation :-**

Visualizations (saved in ./outputs).  
CSV files for top hashtags and n-grams.

**Workflow :-**

**1. Load Dataset :-**  Provide a CSV file (default: twitter_training.csv).

**2. Data Cleaning :-**  Missing values are handled; sentiment labels are normalized to positive, negative, neutral.

**3. Exploratory Analysis :-**  Distribution plots, time-series trends, hashtag frequencies, and n-gram patterns are generated.

**4. Model Training :-**  A logistic regression classifier is trained on TF-IDF features. Performance is summarized with accuracy, precision, recall, and F1-score.

**5. Results Export :-**  Plots and summary tables are stored in the outputs directory.
