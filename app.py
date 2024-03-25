from flask import Flask, render_template, request
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

model = pickle.load(open('logistic_regression.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def clean_txt(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def prediction(input_text):
    cleaned_text = clean_txt(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(model.predict(input_vectorized)[0])
    return predicted_emotion,label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    predicted_emotion, label = prediction(input_text)
    return render_template('result.html', predicted_emotion=predicted_emotion, label=label)

if __name__ == '__main__':
    app.run(debug=True)
