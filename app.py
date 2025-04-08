from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Initialize NLTK components
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load the model, vectorizer, and LDA
model_path = os.path.join('models', 'logistic_regression_model_lda.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
lda_path = os.path.join('models', 'lda_transformer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(lda_path, 'rb') as f:
    lda = pickle.load(f)

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log request details for debugging
        print(f"Request Content-Type: {request.content_type}")
        print(f"Request data: {request.data}")
        
        # Try to get JSON data with silent=True to avoid errors
        data = request.get_json(silent=True)
        
        if data is None:
            return jsonify({
                'error': 'Invalid JSON data. Make sure Content-Type is application/json',
                'received_content_type': request.content_type
            }), 400
        
        if 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'received_fields': list(data.keys())
            }), 400
            
        text = data['text']
        if not text or not isinstance(text, str):
            return jsonify({
                'error': 'Invalid text provided. Text must be a non-empty string',
                'received_type': type(text).__name__
            }), 400
        
        # Single text analysis
        cleaned_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([cleaned_text])
        text_lda = lda.transform(text_vectorized.toarray())
        prediction = model.predict(text_lda)[0]
        probability = model.predict_proba(text_lda)[0].max()
        
        return jsonify({
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(probability)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)