from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
import base64
import io
from PIL import Image
import os
import warnings

# Suppress TensorFlow warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
class_names = {}
IMG_SIZE = 160

def load_model():
    """Load model and class information once at startup with error handling"""
    global model, class_names, IMG_SIZE
    
    try:
        print("🔄 Loading TensorFlow model...")
        
        # Check if model file exists
        if not os.path.exists("plant_disease_model_lite.keras"):
            raise FileNotFoundError("Model file 'plant_disease_model_lite.keras' not found")
        
        # Load TensorFlow model with compatibility settings
        model = tf.keras.models.load_model(
            "plant_disease_model_lite.keras",
            compile=False  # Skip compilation for compatibility
        )
        
        # Recompile model for current TensorFlow version
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model loaded successfully with TensorFlow {tf.__version__}")
        print(f"✅ Model input shape: {model.input_shape}")
        
        # Load class information
        if not os.path.exists("plant_disease_info.json"):
            print("⚠️ Class info file not found, using default classes")
            # Fallback class names if JSON file is missing
            class_names = {str(i): f"Disease_Class_{i}" for i in range(38)}
            IMG_SIZE = 160
        else:
            with open("plant_disease_info.json", 'r') as f:
                disease_info = json.load(f)
            
            class_names = disease_info['classes']
            IMG_SIZE = disease_info.get('input_shape', [160, 160, 3])[0]
        
        print(f"✅ Loaded {len(class_names)} disease classes")
        print(f"✅ Image size set to: {IMG_SIZE}x{IMG_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        model = None
        return False

# Load model when the application starts
print("🚀 Initializing Plant Disease Detection API...")
model_loaded = load_model()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "🌱 Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "endpoints": {
            "health": "/api/health (GET)",
            "predict": "/api/predict (POST)",
            "classes": "/api/classes (GET)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Plant Disease Detection API is running on Render",
        "model_loaded": model is not None,
        "total_classes": len(class_names),
        "image_size": IMG_SIZE,
        "tensorflow_version": tf.__version__,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Disease prediction endpoint with comprehensive error handling"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False, 
                "error": "Model not loaded. Please check server logs.",
                "model_loaded": False
            }), 500

        # Validate file upload
        if 'image' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No image file provided. Use 'image' as the form-data key.",
                "expected_format": "multipart/form-data with 'image' field"
            }), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False, 
                "error": "No file selected"
            }), 400

        # Validate file type
        allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        file_extension = image_file.filename.lower().split('.')[-1]
        if file_extension not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"Unsupported file type: {file_extension}",
                "allowed_types": allowed_extensions
            }), 400

        # Process image
        print(f"📷 Processing image: {image_file.filename}")
        image = Image.open(image_file)
        original_size = image.size
        original_mode = image.mode
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"🔄 Converted from {original_mode} to RGB")
        
        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✅ Image preprocessed: {img_array.shape}")

        # Make prediction
        print("🤖 Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        # Get predicted disease name
        predicted_disease = class_names.get(str(predicted_class_index), f"Unknown_Class_{predicted_class_index}")
        
        # Parse disease information
        disease_parts = predicted_disease.split('___')
        if len(disease_parts) > 1:
            plant_type = disease_parts[0].replace('_', ' ').title()
            disease_name = disease_parts[1].replace('_', ' ').title()
        else:
            plant_type = "Unknown"
            disease_name = predicted_disease.replace('_', ' ').title()
        
        is_healthy = 'healthy' in predicted_disease.lower()

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        for idx in top_3_indices:
            disease = class_names.get(str(idx), f"Class_{idx}")
            conf = float(predictions[0][idx])
            top_predictions.append({
                "disease": disease,
                "confidence": conf,
                "confidence_percent": f"{conf:.1%}"
            })

        print(f"✅ Prediction completed: {predicted_disease} ({confidence:.3f})")

        response_data = {
            "success": True,
            "timestamp": str(tf.timestamp()),
            "prediction": {
                "disease": predicted_disease,
                "plant_type": plant_type,
                "disease_name": disease_name,
                "is_healthy": is_healthy,
                "confidence": confidence,
                "confidence_percent": f"{confidence:.1%}"
            },
            "top_predictions": top_predictions,
            "image_info": {
                "original_size": original_size,
                "original_mode": original_mode,
                "processed_size": [IMG_SIZE, IMG_SIZE],
                "filename": image_file.filename
            },
            "model_info": {
                "tensorflow_version": tf.__version__,
                "total_classes": len(class_names)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"❌ Prediction error: {error_type} - {error_msg}")
        
        return jsonify({
            "success": False, 
            "error": error_msg,
            "error_type": error_type,
            "model_loaded": model is not None
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available disease classes"""
    return jsonify({
        "success": True,
        "total_classes": len(class_names),
        "classes": class_names,
        "model_loaded": model is not None
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for debugging"""
    return jsonify({
        "message": "Test endpoint working",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "available_files": [f for f in os.listdir('.') if f.endswith(('.keras', '.json'))],
        "current_directory": os.getcwd()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/api/health", "/api/predict", "/api/classes", "/api/test"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

# Production-ready configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"🤖 Model loaded: {model is not None}")
    print(f"🐍 Python version: {os.sys.version}")
    print(f"🔢 TensorFlow version: {tf.__version__}")
    
    if not model_loaded:
        print("⚠️  WARNING: Model failed to load. API will return errors for predictions.")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode
    )
