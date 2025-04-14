from flask import Flask, request, jsonify
from utils import news_classifier
from flask_cors import CORS
import os
import logging

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        result = news_classifier.predict(text)
        
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'is_fake': result['is_fake'],
            'detected_patterns': result.get('detected_patterns', []) 
        })
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)