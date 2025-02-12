from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import base64
import re
import io
import joblib
import os
import time

# Load the trained SVM model (replace with the correct path to your model)
MODEL_PATH = './model/svm_digit_classifier_version4.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model file not found at {MODEL_PATH}")

svm_model = joblib.load(MODEL_PATH)

app = Flask(__name__)


# Helper function to process image
def process_image(img):
    # Resize the image to 28x28 like MNIST
    img = img.resize((28, 28)).convert('L')  # Convert to grayscale
    img_np = np.array(img)

    # Normalize the image
    img_np = img_np.astype('float32') / 255

    # Flatten the image to 1D (required by SVM input)
    img_np = img_np.reshape(1, -1)

    return img_np


@app.route('/')
def index():
    return send_from_directory('ui/static', 'index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('ui/static', 'favicon.ico')


# Route to handle prediction from canvas drawing
@app.route('/predict_canvas', methods = ['POST'])
def predict_canvas():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'prediction': 'Error: No image data found.'})

    try:
        img_data = data['image']
        # Decode base64 image
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        img_bytes = io.BytesIO(base64.b64decode(img_str))
        img = Image.open(img_bytes)

        # Process the image and make a prediction
        processed_image = process_image(img)

        start_time = time.time()
        prediction = svm_model.predict(processed_image)
        end_time = time.time()
        # Calculate prediction time
        prediction_time = end_time - start_time

        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_time': prediction_time
        })
    except Exception as e:
        return jsonify({'prediction': f'Error in processing image: {str(e)}'})


# Route to handle image upload (for both image upload and canvas draw scenarios)
@app.route('/upload_image', methods = ['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'prediction': 'Error: No image file uploaded.'})

    try:
        file = request.files['image']
        img = Image.open(file.stream)

        # Process the image and make a prediction
        processed_image = process_image(img)
        start_time = time.time()
        prediction = svm_model.predict(processed_image)
        end_time = time.time()
        # Calculate prediction time
        prediction_time = end_time - start_time

        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_time': prediction_time
        })
    except Exception as e:
        return jsonify({'prediction': f'Error in processing image: {str(e)}'})


# Route to handle basic calculations using predicted operation
@app.route('/calculate', methods = ['POST'])
def calculate():
    try:
        data = request.get_json()
        first_number = int(data['firstNumber'])
        second_number = int(data['secondNumber'])
        operation_image = data['operationImage']

        operation = predict_image(operation_image)

        operations = {
            10: lambda x, y: x + y,  # +
            11: lambda x, y: x - y,  # -
            12: lambda x, y: x * y,  # *
            13: lambda x, y: x / y if y != 0 else 'Error: Division by zero'  # /
        }

        if operation not in operations:
            return jsonify({'result': 'Error: Invalid operation'})

        result = operations[operation](first_number, second_number)
        return jsonify({'result': result})
    except ValueError:
        return jsonify({'result': 'Error: Invalid input'})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'})


# Helper function to process the operation image and predict the operation
def predict_image(image_data):
    try:
        img_bytes = io.BytesIO(base64.b64decode(image_data))
        img = Image.open(img_bytes)

        processed_image = process_image(img)
        prediction = svm_model.predict(processed_image)

        return int(prediction[0])
    except Exception as e:
        raise ValueError(f"Error in prediction: {e}")


if __name__ == '__main__':
    app.run(debug = True)
