"""
Plant Disease Detection System - Flask Backend
Real-time Plant Leaf Disease Detection Using Deep Learning
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Create uploads folder if not exists
os.makedirs("static/uploads", exist_ok=True)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Model configuration
MODEL_PATH = "model/plant_leaf_disease_final_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = 224

# Global variables
model = None
class_names = None
prediction_history = []

# =========================
# YOUR EXISTING DISEASE_INFO
# =========================
# KEEP YOUR FULL DISEASE_INFO DICTIONARY SAME AS BEFORE
# (copy your existing big DISEASE_INFO here)

DEFAULT_DISEASE_INFO = {
    'description': 'Plant disease detected. Consult with agricultural expert for detailed diagnosis.',
    'treatment': 'Remove affected leaves, maintain plant hygiene, apply appropriate treatments.',
    'prevention': 'Regular monitoring, proper spacing, adequate nutrition, and water management.'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    global model, class_names

    try:
        if os.path.exists(MODEL_PATH):
            print("Loading model from:", MODEL_PATH)
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")
        else:
            print("❌ Model file not found:", MODEL_PATH)
            model = None

        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"✅ Loaded {len(class_names)} class names")
        else:
            print("❌ Class names file not found")
            class_names = None

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        class_names = None


def preprocess_image(image_path):
    try:
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_disease(image_path):
    try:
        if model is None:
            print("❌ Model not loaded")
            return None

        if class_names is None:
            print("❌ Class names not loaded")
            return None

        processed_image = preprocess_image(image_path)

        if processed_image is None:
            return None

        predictions = model.predict(processed_image, verbose=0)

        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        disease_name = class_names.get(
            str(predicted_class_idx),
            f"Unknown_Disease_{predicted_class_idx}"
        )

        top_3_idx = np.argsort(predictions[0])[-3:][::-1]

        top_3_predictions = [
            {
                'disease': class_names.get(str(idx), f"Unknown_{idx}"),
                'confidence': float(predictions[0][idx]) * 100
            }
            for idx in top_3_idx
        ]

        return {
            'disease': disease_name,
            'confidence': confidence * 100,
            'top_3': top_3_predictions
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def get_disease_info(disease_name):
    return DISEASE_INFO.get(disease_name, DEFAULT_DISEASE_INFO)


# IMPORTANT: Load model globally for Render / Gunicorn
load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_disease(filepath)

        if prediction is None:
            return jsonify({
                'success': False,
                'error': 'Model prediction failed. Check model loading.'
            }), 500

        disease_info = get_disease_info(prediction['disease'])

        response = {
            'success': True,
            'disease': prediction['disease'],
            'confidence': round(prediction['confidence'], 2),
            'top_predictions': prediction['top_3'],
            'info': disease_info,
            'image_url': f'/static/uploads/{filename}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        prediction_history.append(response)

        if len(prediction_history) > 50:
            prediction_history.pop(0)

        return jsonify(response)

    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    return jsonify({
        'success': True,
        'history': prediction_history[-10:]
    })


@app.route('/clear-history', methods=['POST'])
def clear_history():
    global prediction_history
    prediction_history = []
    return jsonify({
        'success': True,
        'message': 'History cleared'
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes_loaded': class_names is not None
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    print("=" * 70)
    print("PLANT DISEASE DETECTION SYSTEM - STARTING")
    print("=" * 70)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
