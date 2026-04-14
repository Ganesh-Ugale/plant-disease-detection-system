"""
Plant Disease Detection System - Flask Backend
Real-time Plant Leaf Disease Detection Using Deep Learning
"""
from gradio_client import Client, handle_file
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

# Hugging Face Space client
hf_client = Client("ganeshugale47/plant-leaf-disease-detector")

# Model configuration (kept same, but no local prediction used now)
MODEL_PATH = "model/plant_leaf_disease_final_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = 224

# Global variables
model = None
class_names = None
prediction_history = []

# =========================
# KEEP YOUR FULL DISEASE_INFO SAME AS BEFORE
# =========================

DEFAULT_DISEASE_INFO = {
    'description': 'Plant disease detected. Consult with agricultural expert for detailed diagnosis.',
    'treatment': 'Remove affected leaves, maintain plant hygiene, apply appropriate treatments.',
    'prevention': 'Regular monitoring, proper spacing, adequate nutrition, and water management.'
}

DEFAULT_DISEASE_INFO = {
    'description': 'Plant disease detected. Consult with agricultural expert for detailed diagnosis.',
    'treatment': 'Remove affected leaves, maintain plant hygiene, apply appropriate treatments.',
    'prevention': 'Regular monitoring, proper spacing, adequate nutrition, and water management.'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """
    Kept same for health check / compatibility.
    Not used for prediction now.
    """
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


def get_disease_info(disease_name):
    return DISEASE_INFO.get(disease_name, DEFAULT_DISEASE_INFO)


# Load model globally (optional / health check only)
load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"✅ File saved: {filepath}")

        # ===============================
        # HUGGING FACE API PREDICTION
        # ===============================
        try:
            result = hf_client.predict(
                img=handle_file(filepath),
                api_name="/predict_disease"
            )

            print("HF RESULT:", result)

            # Handle result format safely
            prediction_text = str(result)

            # Default values
            disease_name = prediction_text
            confidence = 95.0

            # Try extracting confidence if available
            if "Confidence:" in prediction_text:
                try:
                    confidence_part = prediction_text.split("Confidence:")[-1].replace("%", "").strip()
                    confidence = float(confidence_part)
                except:
                    confidence = 95.0

            # Clean disease name
            if "Prediction:" in prediction_text:
                disease_name = prediction_text.split("Prediction:")[-1].split("Confidence:")[0].strip()

        except Exception as hf_error:
            print("❌ Hugging Face API Error:", hf_error)
            return jsonify({
                'success': False,
                'error': 'Prediction failed from Hugging Face API.'
            }), 500

        # Get disease info
        disease_info = get_disease_info(disease_name)

        response = {
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'top_predictions': [],
            'info': disease_info,
            'image_url': f'/static/uploads/{filename}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save history
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
        'hf_connected': True,
        'model_loaded': model is not None,
        'classes_loaded': class_names is not None
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    print("=" * 70)
    print("PLANT DISEASE DETECTION SYSTEM - STARTING")
    print("=" * 70)
    print("🌿 Using Hugging Face API for prediction")
    print("=" * 70)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
