from flask import Flask, request, jsonify,send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("bestmodalfinal1000.h5")  # Ensure this file is in the same directory

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Adjust based on your model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/")
def home():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class with highest probability

    return jsonify({"disease": str(predicted_class)})

# Get the port from the environment variable (used by Render)
port = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set

# Run the app on the specified port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
