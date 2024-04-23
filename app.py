import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Load the model
model = tf.keras.models.load_model('beef_pork_horse_classifier.h5')

# function to predict image from URL for the Streamlit UI
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

    return {'image': image, 'predicted_class': predicted_class, 'probabilities': probabilities_class}

def main():
    # Streamlit UI
    st.title('Beef, Pork, and Horse Classifier')
    st.markdown('This app predicts whether an image contains beef, pork, or horse.')

    # Input image URL
    url = st.text_input('Enter Image URL:')
    if st.button('Predict'):
        if url:
            result = predict(url)
            if 'error' in result:
                st.error(result['error'])
            else:
                st.image(result['image'], caption='Input Image', use_column_width=True)
                st.success(f'Predicted Class: {result["predicted_class"]}')
                st.success(f'Probabilities: {result["probabilities"]}%')

if __name__ == '__main__':
    main()
