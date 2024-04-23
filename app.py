from flask import Flask, request, jsonify
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('beef_pork_horse_classifier.h5')

# function to predict image from URL for the Streamlit UI
def predict_ui(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return {'error': f'Failed to download image from URL: {str(e)}'}

    # Resize image
    image = image.resize((224, 224))

    # Preprocess image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    probabilities = tf.reduce_max(predictions, axis=1) * 100

    class_names = ['Horse', 'Meat', 'Pork']
    predicted_class = class_names[predicted_label]
    probabilities_class = '%.2f' % probabilities.numpy()[0]

    return {'image': image, 'predicted_class': predicted_class, 'probabilities': probabilities_class}

# function to predict image from URL
def predict(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return {'error': f'Failed to download image from URL: {str(e)}'}

    # Resize image
    image = image.resize((224, 224))

    # Preprocess image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    probabilities = tf.reduce_max(predictions, axis=1) * 100

    class_names = ['Horse', 'Meat', 'Pork']
    predicted_class = class_names[predicted_label]
    probabilities_class = '%.2f' % probabilities.numpy()[0]

    return {'predicted_class': predicted_class, 'probabilities': probabilities_class}

# API Endpoints
@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'models': 'MobileNetV3Large', 'framework': 'TensorFlow', 'task': 'Image Classification for Beef, Pork, and Horse', 'accuracy': '97.43%', 'input': 'URL', 'output': 'Predicted class and probabilities', 'model_url': 'https://www.kaggle.com/code/hafidhmuhammadakbar/mobilenetv3large-fix', 'api_documentation': 'https://github.com/hafidhmuhammadakbar'})

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'url' not in request.form:
        return jsonify({'error': 'No URL provided'})

    url = request.form['url']
    result = predict(url)
    if 'error' in result:
        return jsonify(result), 400

    return jsonify(result)

# Streamlit UI
st.title('Beef, Pork, and Horse Classifier')
st.markdown('This app predicts whether an image contains beef, pork, or horse.')

# Add a redirect hyperlink
api_url = 'http://meat-classifier-api.streamlit.app:5000'
st.markdown(f'[Redirect to API]({api_url})')

# Input image URL
url = st.text_input('Enter Image URL:')
if st.button('Predict'):
    if url:
        result = predict_ui(url)
        if 'error' in result:
            st.error(result['error'])
        else:
            st.image(result['image'], caption='Input Image', use_column_width=True)
            st.success(f'Predicted Class: {result["predicted_class"]}')
            st.success(f'Probabilities: {result["probabilities"]}%')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
