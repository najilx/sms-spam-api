# train_model.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import load_and_prepare_data

# Load and prepare data
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Save the vectorizer and the classifier
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(clf, 'classifier.pkl')

if __name__ == '__main__':
    print("Model training completed.")
