# Importing Necessary Libraries
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Downloading NLTK Data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# File Path
file_path = "/content/354--mail_data.csv"

# Loading the Dataset
df = pd.read_csv(file_path)

# Display Column Names
print("\nColumn Names in Dataset:", df.columns)

# Clean Column Names
df.columns = df.columns.str.strip().str.lower()
print("\nCleaned Column Names:", df.columns)

# Rename Columns
df = df.rename(columns={'category': 'label', 'message': 'text'})

# Check for Necessary Columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise KeyError("Check the column names in your dataset. They should be 'category' and 'message' before renaming.")

# Handle NaN and Non-string Values
df['text'] = df['text'].fillna('').astype(str)

# Handle Unknown Labels and Map to Binary Values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop Rows with NaN in Label
df = df.dropna(subset=['label'])

# Ensure Label Column is Integer Type
df['label'] = df['label'].astype(int)

# Display First 5 Rows
print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Data Cleaning Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove Numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove Punctuation
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stopwords.words('english')]  # Remove Stopwords
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize
    text = ' '.join(text)  # Join Words
    return text

# Applying Cleaning Function
df['cleaned_text'] = df['text'].apply(clean_text)

# Display First 5 Cleaned Texts
print("\nCleaned Text Samples:")
print(df['cleaned_text'].head())

# Splitting Data into Features and Labels
X = df['cleaned_text']
y = df['label']

# Vectorizing Text Data
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model Training with Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Evaluating Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Testing Model with Custom Inputs
def predict_spam(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

# Input Loop for Testing
while True:
    input_text = input("\nEnter a message to classify (or type 'exit' to quit): ")
    if input_text.lower() == 'exit':
        break
    result = predict_spam(input_text)
    print(f"Prediction: {result}")
