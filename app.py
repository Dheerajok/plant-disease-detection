from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
class_names = {}
IMG_SIZE = 160

def load_model():
    """Load model and class information once at startup"""
    global model, class_names, IMG_SIZE
    
    try:
        # Load TensorFlow model
        model = tf.keras.models.load_model("plant_disease_model_lite.keras")
        print("✅ Model loaded successfully")
        
        # Load class information
        with open("plant_disease_info.json", 'r') as f:
            disease_info = json.load(f)
        
        class_names = disease_info['classes']
        IMG_SIZE = disease_info['input_shape'][0]
        print(f"✅ Loaded {len(class_names)} disease classes")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Load model when the application starts
load_model()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "🌱 Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
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
        "image_size": IMG_SIZE
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Disease prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                "success": False, 
                "error": "Model not loaded"
            }), 500

        # Validate file upload
        if 'image' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No image file provided. Use 'image' as the key."
            }), 400

        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "success": False, 
                "error": "No file selected"
            }), 400

        # Process image
        image = Image.open(image_file)
        original_size = image.size
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        predicted_disease = class_names.get(str(predicted_class_index), "Unknown")
        
        # Parse disease information
        disease_parts = predicted_disease.split('___')
        plant_type = disease_parts[0].replace('_', ' ').title() if len(disease_parts) > 1 else "Unknown"
        disease_name = disease_parts[1].replace('_', ' ').title() if len(disease_parts) > 1 else predicted_disease
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
        return jsonify({
            "success": False, 
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available disease classes"""
    return jsonify({
        "success": True,
        "total_classes": len(class_names),
        "classes": class_names
    })

# Production-ready configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode
    )
