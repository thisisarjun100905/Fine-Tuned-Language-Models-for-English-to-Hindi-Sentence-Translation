from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Define the path to the model.h5 file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.h5')

# Load the model and tokenizer
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Since model.h5 is a TensorFlow model, load it using TensorFlow'
pretrained_model_name = 'Helsinki-NLP/opus-mt-en-hi'
model = TFAutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)

# For the tokenizer, specify the correct pre-trained model name or path
  # Replace with your tokenizer's pre-trained model name
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Render an HTML page for input

# Route for translation API
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input. Please provide text.'}), 400

    input_text = data['text']
    
    # Tokenize input text
    tokenized = tokenizer(input_text, return_tensors='tf')
    
    # Generate translation
    translated_tokens = model.generate(**tokenized, max_length=128)
    
    # Decode translation
    with tokenizer.as_target_tokenizer():
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return jsonify({'translation': translated_text})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
