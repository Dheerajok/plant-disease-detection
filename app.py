from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import warnings
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = {}
IMG_SIZE = 160
tf = None

def load_tensorflow():
    """Load TensorFlow with version compatibility"""
    global tf
    try:
        import tensorflow as tf_module
        tf = tf_module
        print(f"✅ TensorFlow {tf.version.VERSION} loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load TensorFlow: {e}")
        return False

def load_model():
    """Load model with fallback options"""
    global model, class_names, IMG_SIZE
    
    if not load_tensorflow():
        return False
    
    try:
        # Method 1: Try loading your trained model
        model_files = [
            "plant_disease_model_lite.keras",
            "plant_disease_model.keras",
            "model.keras"
        ]
        
        model_loaded = False
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    print(f"🔄 Loading model: {model_file}")
                    model = tf.keras.models.load_model(model_file, compile=False)
                    
                    # Recompile for current TensorFlow version
                    model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print(f"✅ Model loaded: {model_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_file}: {e}")
                    continue
        
        if not model_loaded:
            print("⚠️ No model file found, creating dummy model for testing")
            model = create_dummy_model()
        
        # Load class information
        if os.path.exists("plant_disease_info.json"):
            with open("plant_disease_info.json", 'r') as f:
                disease_info = json.load(f)
            class_names = disease_info.get('classes', {})
            IMG_SIZE = disease_info.get('input_shape', [160, 160, 3])[0]
            print(f"✅ Loaded {len(class_names)} classes from JSON")
        else:
            # Default classes for PlantVillage dataset
            class_names = create_default_classes()
            IMG_SIZE = 160
            print(f"✅ Using default {len(class_names)} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in load_model: {e}")
        return False

def create_dummy_model():
    """Create a simple dummy model for testing if main model fails"""
    print("🔄 Creating dummy model for testing...")
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(160, 160, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(38, activation='softmax')
    ])
    dummy_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return dummy_model

def create_default_classes():
    """Create default class names"""
    return {
        "0": "Apple___Apple_scab",
        "1": "Apple___Black_rot",
        "2": "Apple___Cedar_apple_rust",
        "3": "Apple___healthy",
        "4": "Tomato___Bacterial_spot",
        "5": "Tomato___Early_blight",
        "6": "Tomato___Late_blight",
        "7": "Tomato___healthy",
        "8": "Potato___Early_blight",
        "9": "Potato___Late_blight",
        "10": "Potato___healthy",
        # Add more as needed
        **{str(i): f"Disease_Class_{i}" for i in range(11, 38)}
    }

# Initialize on startup
print("🚀 Initializing Plant Disease Detection API...")
model_loaded = load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🌱 Plant Disease Detection API",
        "status": "running",
        "model_loaded": model_loaded,
        "tensorflow_version": tf.version.VERSION if tf else "Not loaded",
        "total_classes": len(class_names),
        "endpoints": {
            "health": "/api/health (GET)",
            "predict": "/api/predict (POST)",
            "classes": "/api/classes (GET)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Plant Disease Detection API is running on Render",
        "model_loaded": model_loaded,
        "tensorflow_version": tf.version.VERSION if tf else "Not loaded",
        "total_classes": len(class_names),
        "image_size": IMG_SIZE,
        "available_files": [f for f in os.listdir('.') if f.endswith(('.keras', '.json'))]
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    try:
        if not model_loaded or model is None:
            return jsonify({
                "success": False, 
                "error": "Model not loaded. Check server logs.",
                "available_endpoints": ["/api/health"]
            }), 500

        if 'image' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No image file provided. Use 'image' as form-data key."
            }), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False, 
                "error": "No file selected"
            }), 400

        # Process image
        print(f"📷 Processing: {image_file.filename}")
        image = Image.open(image_file)
        original_size = list(image.size)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        print("🤖 Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
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
            disease = class_names.get(str(int(idx)), f"Class_{int(idx)}")
            conf = float(predictions[0][idx])
            top_predictions.append({
                "disease": disease,
                "confidence": conf,
                "confidence_percent": f"{conf:.1%}"
            })

        print(f"✅ Prediction: {predicted_disease} ({confidence:.3f})")

        return jsonify({
            "success": True,
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
                "processed_size": [IMG_SIZE, IMG_SIZE],
                "filename": image_file.filename
            }
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction error: {error_msg}")
        
        return jsonify({
            "success": False, 
            "error": error_msg,
            "model_loaded": model_loaded
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({
        "success": True,
        "total_classes": len(class_names),
        "classes": class_names
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check what's available"""
    return jsonify({
        "tensorflow_available": tf is not None,
        "tensorflow_version": tf.version.VERSION if tf else None,
        "model_loaded": model_loaded,
        "files_in_directory": os.listdir('.'),
        "model_files": [f for f in os.listdir('.') if f.endswith('.keras')],
        "json_files": [f for f in os.listdir('.') if f.endswith('.json')],
        "current_directory": os.getcwd()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🤖 Model loaded: {model_loaded}")
    print(f"🔢 TensorFlow: {tf.version.VERSION if tf else 'Not loaded'}")
    print(f"📊 Classes: {len(class_names)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
