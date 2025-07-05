from flask import Flask, request, jsonify, send_from_directory
from scipy.ndimage import affine_transform, shift, center_of_mass
from PIL import Image, ImageOps
import numpy as np
import base64
import re
import io
import joblib
import os
import time

# Load the trained SVM model (replace with the correct path to your model)
MODEL_PATH = "D:\Graduation_Project\model\svm_digit_classifier_version5.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model file not found at {MODEL_PATH}")

svm_model = joblib.load(MODEL_PATH)

app = Flask(__name__)


# # Help align handwritten digits (which may be skewed due to drawing method).
# Giúp căn chỉnh chữ số vẽ tay (có thể bị nghiêng do cách vẽ).
def deskew(img_np):
    c0, c1 = np.mgrid[:28, :28]
    total = img_np.sum() + 1e-5
    x_c = (c1 * img_np).sum() / total
    y_c = (c0 * img_np).sum() / total
    x = c1 - x_c
    y = c0 - y_c
    mu11 = (x * y * img_np).sum() / total
    mu02 = (y ** 2 * img_np).sum() / total
    alpha = mu11 / (mu02 + 1e-5)
    matrix = np.array([[1, alpha, -0.5 * 28 * alpha],
                       [0, 1, 0]])
    return affine_transform(img_np, matrix, order = 1, mode = 'constant', cval = 0)


# Shift the image so the center of mass of the digit is at the center.
# Dịch chuyển hình ảnh để trọng tâm của chữ số nằm chính giữa.
def center_image(img_np):
    cy, cx = center_of_mass(img_np)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    return shift(img_np, [shifty, shiftx], mode = 'constant', cval = 0)


# Helper function to process canvas image
def process_image_canvas(img):
    # 1. Grayscale
    img = img.convert('L')

    # 2. Invert colors if drawing is black on white
    img = ImageOps.invert(img)

    # 3. Convert to numpy and initial threshold
    arr = np.array(img)
    arr = (arr > 50) * 255

    # 4. Crop bounding box
    ys, xs = np.where(arr)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        arr = arr[y0:y1 + 1, x0:x1 + 1]
    # else, keep full array

    # 5. Resize to max 20×20, maintain aspect
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    img_crop = Image.fromarray(arr.astype('uint8'))
    img_resized = img_crop.resize((new_w, new_h), Image.LANCZOS)

    # 6. Pad to 28×28
    new_img = Image.new('L', (28, 28), color = 0)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    new_img.paste(img_resized, (x_off, y_off))

    new_img.save("processed_image.png")

    # 7. Final threshold to get crisp pixels
    arr_final = np.array(new_img)
    arr_final = (arr_final > 128) * 255

    # 8. Deskew and center
    arr_deskewed = deskew(arr_final)
    arr_centered = center_image(arr_deskewed)

    # 9. Normalize
    arr_norm = arr_centered.astype('float32') / 255.0
    return arr_norm.reshape(1, -1)


# Helper function to process upload image
def process_image_upload(img):
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
        processed_image = process_image_canvas(img)

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


# Route to handle prediction for image file upload
@app.route('/predict_image', methods = ['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'prediction': 'Error: No image file uploaded.'})

    try:
        file = request.files['file']
        img = Image.open(file.stream)

        # Process the image and make a prediction
        processed_image = process_image_upload(img)
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
        first = int(data['firstNumber'])
        second = int(data['secondNumber'])
        op = data['operationImage']
        operations = {
            10: lambda x, y: x + y,
            11: lambda x, y: x - y,
            12: lambda x, y: x * y,
            13: lambda x, y: x / y if y != 0 else 'Error: Division by zero'
        }
        if op not in operations:
            return jsonify({'result': 'Error: Invalid operation'})
        return jsonify({'result': operations[op](first, second)})
    except Exception as e:
        return jsonify({'result': f'Error: {e}'})


if __name__ == '__main__':
    app.run(debug = True)
