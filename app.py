"""
Plant Disease Detection System - Flask Backend
Real-time Plant Leaf Disease Detection Using Deep Learning
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Initialize Flask app
app = Flask(__name__)

# Create uploads folder if not exists
os.makedirs("static/uploads", exist_ok=True)
# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Model configuration
MODEL_PATH = "model/plant_leaf_disease_final_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = 224  # EfficientNetB3 input size

# Global variables
model = None
class_names = None
prediction_history = []

# Disease treatment information (you can expand this)
DISEASE_INFO = {

    # ================= APPLE =================
    'Apple___Apple_scab': {
        'description': 'Fungal disease causing dark scab-like spots on apple leaves and fruits.',
        'treatment': 'Apply fungicides like captan or mancozeb, remove infected leaves.',
        'prevention': 'Plant resistant varieties, ensure proper pruning and airflow.'
    },
    'Apple___Black_rot': {
        'description': 'Fungal disease causing black circular lesions on leaves and fruit rot.',
        'treatment': 'Remove infected fruits, apply fungicides.',
        'prevention': 'Sanitation and pruning of dead branches.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Rust disease causing yellow-orange spots on leaves.',
        'treatment': 'Apply rust fungicides, remove nearby juniper plants.',
        'prevention': 'Grow resistant apple varieties.'
    },
    'Apple___healthy': {
        'description': 'Healthy apple leaf with no visible disease.',
        'treatment': 'No treatment required.',
        'prevention': 'Maintain regular watering and fertilization.'
    },

    # ================= BLUEBERRY =================
    'Blueberry___healthy': {
        'description': 'Healthy blueberry leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Maintain soil acidity and proper irrigation.'
    },

    # ================= CHERRY =================
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'White powdery fungal growth on cherry leaves.',
        'treatment': 'Apply sulfur or neem oil sprays.',
        'prevention': 'Avoid overcrowding and improve air circulation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'Healthy cherry leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Regular orchard maintenance.'
    },

    # ================= CORN =================
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Gray rectangular lesions on maize leaves.',
        'treatment': 'Use fungicides and resistant hybrids.',
        'prevention': 'Crop rotation and residue management.'
    },
    'Corn_(maize)___Common_rust': {
        'description': 'Rust-colored pustules on maize leaves.',
        'treatment': 'Apply fungicides if severe.',
        'prevention': 'Grow rust-resistant varieties.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Long gray-green lesions on maize leaves.',
        'treatment': 'Apply fungicides and remove infected debris.',
        'prevention': 'Crop rotation and resistant hybrids.'
    },
    'Corn_(maize)___healthy': {
        'description': 'Healthy maize leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Balanced fertilization.'
    },

    # ================= GRAPE =================
    'Grape___Black_rot': {
        'description': 'Black lesions on grape leaves and fruit.',
        'treatment': 'Apply fungicides like myclobutanil.',
        'prevention': 'Prune vines and remove infected fruit.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Dark spots and tiger stripe patterns on leaves.',
        'treatment': 'Remove infected vines.',
        'prevention': 'Avoid vine stress and wounds.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Brown irregular spots on grape leaves.',
        'treatment': 'Apply fungicides.',
        'prevention': 'Ensure proper vineyard sanitation.'
    },
    'Grape___healthy': {
        'description': 'Healthy grape leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Proper pruning and irrigation.'
    },

    # ================= ORANGE =================
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Serious bacterial disease causing yellow mottling.',
        'treatment': 'Remove infected trees.',
        'prevention': 'Control insect vectors like psyllids.'
    },

    # ================= PEACH =================
    'Peach___Bacterial_spot': {
        'description': 'Dark spots on peach leaves and fruits.',
        'treatment': 'Copper-based sprays.',
        'prevention': 'Use resistant varieties.'
    },
    'Peach___healthy': {
        'description': 'Healthy peach leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Good orchard practices.'
    },

    # ================= PEPPER =================
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Water-soaked spots on pepper leaves.',
        'treatment': 'Copper fungicides.',
        'prevention': 'Use disease-free seeds.'
    },
    'Pepper,_bell___healthy': {
        'description': 'Healthy bell pepper leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Maintain proper spacing.'
    },

    # ================= POTATO =================
    'Potato___Early_blight': {
        'description': 'Brown spots with concentric rings on leaves.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Crop rotation.'
    },
    'Potato___Late_blight': {
        'description': 'Dark brown patches on leaves and stems.',
        'treatment': 'Use copper fungicides.',
        'prevention': 'Avoid wet foliage.'
    },
    'Potato___healthy': {
        'description': 'Healthy potato leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Balanced nutrient supply.'
    },

    # ================= RASPBERRY =================
    'Raspberry___healthy': {
        'description': 'Healthy raspberry leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Regular pruning.'
    },

    # ================= SOYBEAN =================
    'Soybean___healthy': {
        'description': 'Healthy soybean leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Crop rotation.'
    },

    # ================= SQUASH =================
    'Squash___Powdery_mildew': {
        'description': 'White powdery spots on squash leaves.',
        'treatment': 'Neem oil or sulfur sprays.',
        'prevention': 'Improve air circulation.'
    },

    # ================= STRAWBERRY =================
    'Strawberry___Leaf_scorch': {
        'description': 'Dark purple spots on strawberry leaves.',
        'treatment': 'Remove infected leaves.',
        'prevention': 'Avoid overhead watering.'
    },
    'Strawberry___healthy': {
        'description': 'Healthy strawberry leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Good drainage.'
    },

    # ================= TOMATO =================
    'Tomato___Bacterial_spot': {
        'description': 'Small dark spots on tomato leaves.',
        'treatment': 'Copper sprays.',
        'prevention': 'Use certified seeds.'
    },
    'Tomato___Early_blight': {
        'description': 'Brown concentric rings on leaves.',
        'treatment': 'Fungicide application.',
        'prevention': 'Crop rotation.'
    },
    'Tomato___Late_blight': {
        'description': 'Dark greasy lesions on leaves.',
        'treatment': 'Copper-based fungicides.',
        'prevention': 'Avoid wet foliage.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Yellow spots on upper leaf surface.',
        'treatment': 'Improve ventilation.',
        'prevention': 'Reduce humidity.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Small round spots with gray centers.',
        'treatment': 'Remove infected leaves.',
        'prevention': 'Mulching.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Yellow stippling due to mite infestation.',
        'treatment': 'Use miticides.',
        'prevention': 'Maintain humidity.'
    },
    'Tomato___Target_Spot': {
        'description': 'Brown spots with concentric rings.',
        'treatment': 'Apply fungicides.',
        'prevention': 'Remove infected debris.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Leaf curling and yellowing.',
        'treatment': 'Remove infected plants.',
        'prevention': 'Control whiteflies.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Mosaic pattern on leaves.',
        'treatment': 'No cure, remove infected plants.',
        'prevention': 'Disinfect tools.'
    },
    'Tomato___healthy': {
        'description': 'Healthy tomato leaf.',
        'treatment': 'No treatment required.',
        'prevention': 'Proper care.'
    }
}


# Default disease info template
DEFAULT_DISEASE_INFO = {
    'description': 'Plant disease detected. Consult with agricultural expert for detailed diagnosis.',
    'treatment': 'Remove affected leaves, maintain plant hygiene, apply appropriate treatments.',
    'prevention': 'Regular monitoring, proper spacing, adequate nutrition, and water management.'
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the trained model"""
    global model, class_names
    
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading model from:", MODEL_PATH)
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model file not found. Using dummy mode.")
            model = None
        
        # Load class names
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"✅ Loaded {len(class_names)} class names")
        else:
            print("⚠️ Class names file not found. Using dummy classes.")
            class_names = {str(i): f"Disease_Class_{i}" for i in range(38)}
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        class_names = {str(i): f"Disease_Class_{i}" for i in range(38)}


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_disease(image_path):
    """Predict disease from image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return None
        
        # Make prediction
        if model is not None:
            predictions = model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            disease_name = class_names.get(str(predicted_class_idx), f"Unknown_Disease_{predicted_class_idx}")
            
            # Get top 3 predictions
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
        else:
            # Dummy prediction for testing without model
            return {
                'disease': 'Tomato___Late_blight',
                'confidence': 95.5,
                'top_3': [
                    {'disease': 'Tomato___Late_blight', 'confidence': 95.5},
                    {'disease': 'Tomato___Early_blight', 'confidence': 3.2},
                    {'disease': 'Tomato___healthy', 'confidence': 1.3}
                ]
            }
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def get_disease_info(disease_name):
    """Get disease information"""
    return DISEASE_INFO.get(disease_name, DEFAULT_DISEASE_INFO)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict disease
        prediction = predict_disease(filepath)
        
        if prediction is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Get disease information
        disease_info = get_disease_info(prediction['disease'])
        
        # Prepare response
        response = {
            'success': True,
            'disease': prediction['disease'],
            'confidence': round(prediction['confidence'], 2),
            'top_predictions': prediction['top_3'],
            'info': disease_info,
            'image_url': f'/static/uploads/{filename}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to history
        prediction_history.append(response)
        
        # Keep only last 50 predictions
        if len(prediction_history) > 50:
            prediction_history.pop(0)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Get prediction history"""
    return jsonify({
        'success': True,
        'history': prediction_history[-10:]  # Last 10 predictions
    })


@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global prediction_history
    prediction_history = []
    return jsonify({'success': True, 'message': 'History cleared'})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0
    })


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model
    print("="*70)
    print("PLANT DISEASE DETECTION SYSTEM - STARTING")
    print("="*70)
    load_model()
    
    # Start Flask app
    print("\n🌿 Starting Flask application...")
    print("📡 Server running at: http://localhost:5000")
    print("🛑 Press CTRL+C to stop\n")
    print("="*70)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
