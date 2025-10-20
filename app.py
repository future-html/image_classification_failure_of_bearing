"""
Advanced Bearing Defect Detection System using CNN
Detects surface defects, cracks, and wear patterns in mechanical bearings
Uses Convolutional Neural Networks for image classification
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import io
from PIL import Image

app = Flask(__name__)

class BearingDefectCNN:
    def __init__(self):
        self.model = self.build_model()
        self.img_size = (224, 224)
        self.classes = ['Normal', 'Inner_Race_Defect', 'Outer_Race_Defect',
                       'Ball_Defect', 'Cage_Defect']

    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def preprocess_image(self, img):
        # Enhanced preprocessing for bearing surface analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Edge detection for defect boundaries
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological operations to close gaps
        kernel = np.ones((3,3), np.uint8)
        morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Convert back to 3 channels
        processed = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        processed = cv2.resize(processed, self.img_size)

        return processed / 255.0

    def extract_features(self, img):
        # Surface roughness analysis using texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate roughness parameters
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_features = hist.flatten()

        # Frequency domain analysis using FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'roughness_index': float(std_intensity / (mean_intensity + 1e-5)),
            'fft_peak': float(np.max(magnitude))
        }

detector = BearingDefectCNN()

@app.route('/analyze_bearing', methods=['POST'])
def analyze_bearing():
  
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess and predict
    processed = detector.preprocess_image(img)
    processed_batch = np.expand_dims(processed, axis=0)

    predictions = detector.model.predict(processed_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    # Extract surface features
    features = detector.extract_features(img)

    result = {
        'defect_type': detector.classes[predicted_class],
        'confidence': confidence,
        'all_probabilities': {
            detector.classes[i]: float(predictions[0][i])
            for i in range(len(detector.classes))
        },
        'surface_analysis': features,
        'recommendation': generate_maintenance_recommendation(
            detector.classes[predicted_class], confidence, features
        )
    }

    return jsonify(result)

def generate_maintenance_recommendation(defect_type, confidence, features):
    recommendations = {
        'Normal': 'Continue regular monitoring. Schedule next inspection in 3 months.',
        'Inner_Race_Defect': 'CRITICAL: Replace bearing immediately. Risk of catastrophic failure.',
        'Outer_Race_Defect': 'Schedule replacement within 2 weeks. Monitor vibration levels.',
        'Ball_Defect': 'Replace within 1 month. Increase lubrication frequency.',
        'Cage_Defect': 'Replace within 2 weeks. Check alignment and load distribution.'
    }

    base_rec = recommendations.get(defect_type, 'Unknown defect type')

    if features['roughness_index'] > 0.5:
        base_rec += ' High surface roughness detected - improve lubrication.'

    return base_rec

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'operational', 'model': 'CNN', 'version': '1.0'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

# comment something 