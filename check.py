import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model("./models/mango.h5")

# Define the class names
CLASS_NAMES = ['Anthracnose', 'Bacterial_Canker', 'Cutting_Weevil', 'Die_Back', 'Gall_Midge', 'Healthy', 'Powdery_Mildew', 'Sooty_Mould']

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # Adjust target_size to match the model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict_class(image_path, model, class_names):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

# Replace "path_to_your_image.jpg" with the path to your input image
image_path = "D:\Mango Leaf Disease Prediction\data\dataset\\train\Bacterial_Canker\IMG_20211106_120947 (Custom).jpg"
predicted_class, confidence = predict_class(image_path, model, CLASS_NAMES)

# Print the results
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence}")
