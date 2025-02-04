import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (Assume a CSV file with 'review' and 'sentiment' columns)
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv('your_dataset.csv')

# Check if dataset contains the required columns
if 'review' not in data.columns or 'sentiment' not in data.columns:
    raise ValueError("Dataset must contain 'review' and 'sentiment' columns")

# Separate reviews and sentiments
reviews = data['review']
sentiments = data['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Use CountVectorizer for feature extraction
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict sentiments on the test set
predictions = classifier.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")

# Example usage with a new product review
new_review = ["I'm really happy with this purchase. It exceeded my expectations!"]
new_review_vectorized = vectorizer.transform(new_review)
predicted_sentiment = classifier.predict(new_review_vectorized)[0]

print(f"Predicted Sentiment: {predicted_sentiment}")
