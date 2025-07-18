import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (you can download the SMS Spam Collection dataset from UCI Machine Learning Repository)
# Example dataset CSV format:
# label,text
# ham,"Hi, how are you?"
# spam,"You have won $1000! Claim now."


# Replace 'spam.csv' with the path to your dataset
data = pd.read_csv("C:\\Users\\Arshia Garg\\Downloads\\SMSSpamCollectionlabel - SMSspamcoll - SMSSpamCollection.csv", encoding='latin-1')
data.columns = data.columns.str.strip()
data = data.rename(columns={'v1': 'label', 'v2': 'text'})
data = data[['label', 'text']]

# Convert labels to binary (spam = 1, ham = 0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split data into features (X) and labels (y)
X = data['text']
y = data['label']

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test with a new SMS
def predict_sms(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

new_sms = "hello you woni 1000 rs"
print(f"Message: '{new_sms}' is classified as: {predict_sms(new_sms)}")   