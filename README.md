
# Handwritten Digit Recognition System

![Static Badge](https://img.shields.io/badge/Python-3.8-grey?logo=python)
![Static Badge](https://img.shields.io/badge/Numpy-grey?logo=numpy)
![Static Badge](https://img.shields.io/badge/SVM-grey?logo=svm)
![Static Badge](https://img.shields.io/badge/MNIST-grey?logo=mnist)

## Proposal of a Subject: Handwritten Digit Recognition System

### Project Overview

Handwritten digit recognition is a prevalent problem within the field of Optical Character Recognition (OCR) and has a wide array of practical applications, such as digitizing documents, automating data entry, and processing handwritten forms like invoices or surveys.

This project focuses on building a system that can recognize handwritten digits from images or handwritten digits entered on a screen. Additionally, the system can perform basic mathematical operations (addition, subtraction, multiplication, and division) based on the recognized digits.

### Objectives
- Develop a system capable of accurately recognizing handwritten digits (0–9) from input images.
- Convert handwritten inputs into a machine-readable format and allow the system to perform basic mathematical calculations such as addition, subtraction, multiplication, and division on the recognized digits.

### Main Tasks
1. **Data Collection:**
   - Use the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits (0-9), each 28x28 pixels in size.
   
2. **Image Preprocessing:**
   - Normalize the image data (convert pixel values to a scale from 0 to 1).
   - Apply noise reduction techniques to improve recognition accuracy.

3. **Model Development:**
   - Implement a Support Vector Machine (SVM) classifier using a one-vs-all strategy to distinguish between different digits.
   - Train the model on the MNIST dataset and fine-tune its performance.

4. **Evaluation:**
   - Evaluate the model’s performance on the test dataset using metrics such as accuracy, precision, recall, and F1 score.

5. **Application:**
   - Build a user interface where users can upload or draw digits, and the system will return the predicted digit and perform basic calculations.

---

### Dataset Information

- **Dataset:** MNIST
- **Training Images:** 60,000 (28x28 grayscale images)
- **Testing Images:** 10,000 (28x28 grayscale images)
- **Classes:** Digits 0 through 9

---

### Model Implementation

This project uses a Support Vector Machine (SVM) classifier with a "one-vs-all" strategy. The model is manually implemented in Python, without relying on scikit-learn's SVC. PCA is applied for dimensionality reduction, and the data is normalized for optimal performance.

**Steps:**
1. Preprocess the data by normalizing and reducing dimensionality.
2. Train the SVM classifier for each digit (0–9).
3. Evaluate the model's performance using metrics like accuracy, precision, recall, and F1 score.

---

### User Interface

The application provides:
- A field to upload an image containing handwritten digits.
- A canvas for users to draw digits directly.
- An additional input field for entering another number.
- A feature that allows basic mathematical operations (addition, subtraction, multiplication, and division) based on the recognized digits.

---

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/trinhminhds/Graduation_Project.git
   cd Graduation_Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the MNIST dataset (if not done automatically).

4. Run the application:
   ```bash
   python app.py
   ```

---

### Evaluation Metrics

- **Accuracy:** Measures the percentage of correctly classified digits.
- **Precision, Recall, and F1 Score:** Used to assess the model's performance for each digit class.

---

### Future Work

- Improve recognition accuracy by experimenting with different kernels for the SVM.
- Implement more advanced noise reduction techniques.
- Expand the system to recognize alphanumeric characters.
