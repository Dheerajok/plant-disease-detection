import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
@st.cache_resource
def load_model():
    """Load model and classes with caching"""
    try:
        # Try to load your model
        if os.path.exists("plant_disease_model_lite.keras"):
            model = tf.keras.models.load_model("plant_disease_model_lite.keras", compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            st.success("✅ Model loaded successfully!")
            
            # Load class information
            if os.path.exists("plant_disease_info.json"):
                with open("plant_disease_info.json", 'r') as f:
                    disease_info = json.load(f)
                class_names = disease_info['classes']
                img_size = disease_info.get('input_shape', [160, 160, 3])[0]
            else:
                # Default classes
                class_names = {
                    "0": "Apple___Apple_scab", "1": "Apple___Black_rot", 
                    "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy",
                    "4": "Tomato___Bacterial_spot", "5": "Tomato___Early_blight",
                    "6": "Tomato___Late_blight", "7": "Tomato___healthy",
                    "8": "Potato___Early_blight", "9": "Potato___Late_blight", 
                    "10": "Potato___healthy"
                }
                img_size = 160
            
            return model, class_names, img_size
        else:
            st.warning("⚠️ Model file not found, using mock predictions")
            return None, {}, 160
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, {}, 160

def predict_disease(image, model, class_names, img_size):
    """Make prediction on image"""
    if model is None:
        # Mock prediction for demo
        diseases = ["Apple___Apple_scab", "Tomato___Early_blight", "Potato___Late_blight", "Apple___healthy"]
        predicted = np.random.choice(diseases)
        confidence = np.random.uniform(0.75, 0.95)
        
        return predicted, confidence
    
    try:
        # Process image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((img_size, img_size))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        predicted_disease = class_names.get(str(predicted_class_index), "Unknown")
        
        return predicted_disease, confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return "Error", 0.0

def parse_disease_info(predicted_disease, confidence):
    """Parse disease information"""
    disease_parts = predicted_disease.split('___')
    if len(disease_parts) > 1:
        plant_type = disease_parts[0].replace('_', ' ').title()
        disease_name = disease_parts[1].replace('_', ' ').title()
    else:
        plant_type = "Unknown"
        disease_name = predicted_disease.replace('_', ' ').title()
    
    is_healthy = 'healthy' in predicted_disease.lower()
    
    return plant_type, disease_name, is_healthy

# Main app
def main():
    # Header
    st.title("🌱 Plant Disease Detection AI")
    st.markdown("Upload an image of a plant leaf to detect diseases using advanced machine learning.")
    
    # Load model
    model, class_names, img_size = load_model()
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    if model is not None:
        st.sidebar.success("✅ AI Model: Loaded")
        st.sidebar.info(f"📈 Classes: {len(class_names)}")
        st.sidebar.info(f"🖼️ Image Size: {img_size}x{img_size}")
    else:
        st.sidebar.warning("⚠️ Using Demo Mode")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📷 Choose a plant leaf image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Uploaded plant image", use_column_width=True)
            
            # Image info
            st.info(f"**Size:** {image.size[0]} × {image.size[1]}")
            st.info(f"**Mode:** {image.mode}")
            st.info(f"**Format:** {image.format}")
        
        with col2:
            st.subheader("🔬 Analysis Results")
            
            # Prediction button
            if st.button("🔍 Analyze Plant Disease", type="primary"):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Make prediction
                    predicted_disease, confidence = predict_disease(image, model, class_names, img_size)
                    
                    # Parse results
                    plant_type, disease_name, is_healthy = parse_disease_info(predicted_disease, confidence)
                    
                    # Display results
                    if is_healthy:
                        st.success("🌱 **Healthy Plant Detected!**")
                        st.balloons()
                    else:
                        st.error("🦠 **Disease Detected!**")
                    
                    # Result details
                    col1_result, col2_result = st.columns([1, 1])
                    
                    with col1_result:
                        st.metric("🌾 Plant Type", plant_type)
                        st.metric("🦠 Condition", disease_name)
                    
                    with col2_result:
                        st.metric("📊 Confidence", f"{confidence:.1%}")
                        
                        # Confidence bar
                        st.progress(confidence)
                    
                    # Treatment recommendations
                    st.subheader("💊 Recommendations")
                    
                    if is_healthy:
                        st.success("""
                        **🌱 Your plant looks healthy!**
                        
                        **Care Tips:**
                        - Continue regular watering schedule
                        - Ensure adequate sunlight
                        - Monitor for any changes
                        - Maintain good soil drainage
                        """)
                    else:
                        st.warning(f"""
                        **🩺 Treatment for {disease_name}:**
                        
                        **Immediate Actions:**
                        - Consult with agricultural expert
                        - Remove affected plant parts if possible
                        - Apply appropriate fungicide treatment
                        - Improve air circulation around plants
                        - Avoid overhead watering
                        
                        **Prevention:**
                        - Use disease-resistant varieties
                        - Maintain proper plant spacing
                        - Regular monitoring and inspection
                        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**🚀 Powered by TensorFlow & Streamlit** | Built for farmers and agricultural professionals")

if __name__ == "__main__":
    main()
