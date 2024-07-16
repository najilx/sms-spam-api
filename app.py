# app.py
from flask import Flask, request, jsonify
import joblib

# Load the vectorizer and the classifier
vectorizer = joblib.load('vectorizer.pkl')
clf = joblib.load('classifier.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    message_tfidf = vectorizer.transform([message])
    prediction = clf.predict(message_tfidf)[0]
    return jsonify({'prediction': 'spam' if prediction == 1 else 'ham'})

if __name__ == '__main__':
    app.run(debug=True)
