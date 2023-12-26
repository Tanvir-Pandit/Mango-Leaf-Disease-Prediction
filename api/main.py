from flask import Flask, jsonify, request
from werkzeug.utils import redirect
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

MODEL_PATH = "../models/mango.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    'Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil', 'Die_Back',
    'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould'
]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((256, 256))  # Adjust size to match the model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict_class(image_path, model, class_names):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

@app.route('/')
def root():
    return redirect('/apidocs/')

@app.route('/predict_mango', methods=['POST'])
def predict_endpoint():
    """
    Mango Disease Classification API
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file for Mango disease classification
    responses:
      200:
        description: Classification result
        schema:
          properties:
            class:
              type: string
              description: Predicted class
            confidence:
              type: number
              description: Confidence level in percentage
    """
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'File parameter is missing.'}), 400

    image_path = BytesIO(file.read())  # Use BytesIO to read the file stream
    predicted_class, confidence = predict_class(image_path, model, CLASS_NAMES)

    return jsonify({'class': predicted_class, 'confidence': float(confidence) * 100})

if __name__ == '__main__':
    app.run(debug=True)
