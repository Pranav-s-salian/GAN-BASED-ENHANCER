import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


st.set_page_config(
    page_title="Satellite Image Enhancement",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


WEATHER_TYPES = {
    0: "Clear",
    1: "Cloudy", 
    2: "Rainy",
    3: "Foggy"
}


@st.cache_resource
def load_models():
    """
    Load all pre-trained models
    """
    try:
        print("Loading all the created models...")
        
        generator = load_model('weather_aware_generator.h5')
        investigator = load_model('weather_aware_discriminator.h5') 
        classifier = load_model('weather_classifier_updated.h5')
        temporal_model = load_model('temporal_model.h5')
        
        print("Models loaded successfully!")
        
        return {
            'generator': generator,
            'investigator': investigator,
            'classifier': classifier,
            'temporal_model': temporal_model
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None



def preprocess_image(image):
    """
    Preprocess uploaded image for model input
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 256x256
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1] range
    img_array = img_array / 255.0
    
    return img_array

def enhance_satellite_image(weather_image, models, weather_type=None):
    """
    Main function that takes a weather-affected image and returns clear image
    """
    classifier = models['classifier']
    generator = models['generator']
    temporal_model = models['temporal_model']
    
    # If weather type not provided, classify it
    if weather_type is None:
        weather_pred = classifier.predict(weather_image.reshape(1, 256, 256, 3), verbose=0)
        weather_type = np.argmax(weather_pred)
        weather_confidence = np.max(weather_pred)
    else:
        weather_confidence = 1.0  # If manually specified
    
    # One-hot encode weather type
    weather_onehot = keras.utils.to_categorical([weather_type], 4)
    
    # Generate clear image using the generator
    clear_image = generator.predict([weather_image.reshape(1, 256, 256, 3), weather_onehot], verbose=0)
    
    return clear_image[0], weather_type, weather_confidence

def evaluate_with_discriminator(image, models):
    """
    Evaluate image quality using the discriminator/investigator
    """
    investigator = models['investigator']
    
    # Prepare image for discriminator
    img_input = image.reshape(1, 256, 256, 3)
    
    # Get quality score
    quality_score = investigator.predict(img_input, verbose=0)[0][0]
    
    return quality_score


def main():
    # Title and description
    st.title("🛰️ Satellite Image Enhancement System")
    st.markdown("Upload a weather-affected satellite image to get a clear, enhanced version!")
    
    # Load models
    with st.spinner("Loading pre-trained models..."):
        models = load_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please ensure all model files are in the current directory:")
        st.code("""
        - weather_aware_generator.h5
        - weather_aware_discriminator.h5  
        - weather_classifier_updated.h5
        - temporal_model.h5
        """)
        return
    
    st.success("✅ All models loaded successfully!")
    
    # Create main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Weather-Affected Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a satellite image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a weather-affected satellite image (will be resized to 256x256)"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            original_image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(original_image, caption=f"Original size: {original_image.size}", use_column_width=True)
            
            # Process image
            processed_image = preprocess_image(original_image)
            
            # Display processed image
            st.subheader("Processed Image (256x256)")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(processed_image)
            ax.axis('off')
            ax.set_title('Resized to 256x256')
            st.pyplot(fig)
            plt.close()
            
            # Weather classification options
            st.subheader("🌤️ Weather Classification")
            
            auto_classify = st.checkbox("Auto-detect weather condition", value=True)
            manual_weather_type = None
            
            if not auto_classify:
                manual_weather_type = st.selectbox(
                    "Select weather condition:",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: WEATHER_TYPES[x]
                )
            
            # Enhancement button
            if st.button("🚀 Enhance Image", type="primary"):
                with st.spinner("Enhancing satellite image..."):
                    # Enhance the image
                    enhanced_image, detected_weather, confidence = enhance_satellite_image(
                        processed_image, models, manual_weather_type
                    )
                
                # Store results in session state for display in col2
                st.session_state.enhanced_image = enhanced_image
                st.session_state.detected_weather = detected_weather
                st.session_state.weather_confidence = confidence
                st.session_state.original_processed = processed_image
    
    with col2:
        st.header("✨ Enhanced Results")
        
        if hasattr(st.session_state, 'enhanced_image'):
            # Display weather detection results
            st.subheader("🔍 Weather Analysis")
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.metric(
                    "Detected Weather", 
                    WEATHER_TYPES[st.session_state.detected_weather]
                )
            with col_b:
                st.metric(
                    "Confidence", 
                    f"{st.session_state.weather_confidence:.2%}"
                )
            
            # Display enhanced image
            st.subheader("Enhanced Image")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(st.session_state.enhanced_image)
            ax.axis('off')
            ax.set_title('Weather-Cleared Satellite Image')
            st.pyplot(fig)
            plt.close()
            
            # Side-by-side comparison
            st.subheader("📊 Before vs After Comparison")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(st.session_state.original_processed)
            ax1.set_title('Original (Weather-Affected)')
            ax1.axis('off')
            
            ax2.imshow(st.session_state.enhanced_image)
            ax2.set_title('Enhanced (Clear)')
            ax2.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Quality evaluation
            st.subheader("🎯 Image Quality Assessment")
            
            with st.spinner("Evaluating image quality..."):
                original_quality = evaluate_with_discriminator(st.session_state.original_processed, models)
                enhanced_quality = evaluate_with_discriminator(st.session_state.enhanced_image, models)
            
            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Original Quality", f"{original_quality:.3f}")
                st.progress(float(original_quality))
            
            with col_d:
                st.metric("Enhanced Quality", f"{enhanced_quality:.3f}")
                st.progress(float(enhanced_quality))
                
            # Improvement indicator
            improvement = enhanced_quality - original_quality
            if improvement > 0:
                st.success(f"✅ Quality improved by {improvement:.3f} points!")
            elif improvement < 0:
                st.warning(f"⚠️ Quality decreased by {abs(improvement):.3f} points")
            else:
                st.info("📊 Quality remained the same")
            
            # Download button for enhanced image
            st.subheader("💾 Download Enhanced Image")
            
            # Convert enhanced image to PIL format for download
            enhanced_pil = Image.fromarray((st.session_state.enhanced_image * 255).astype(np.uint8))
            
            # Convert PIL image to bytes
            img_buffer = io.BytesIO()
            enhanced_pil.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Enhanced Image",
                data=img_bytes,
                file_name="enhanced_satellite_image.png",
                mime="image/png"
            )
        
        else:
            st.info("👆 Upload an image and click 'Enhance Image' to see results here!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This System")
        st.markdown("""
        This system uses **4 pre-trained models**:
        
        🔍 **Weather Classifier**: Detects weather conditions
        
        🎨 **Generator**: Creates clear images from weather-affected ones
        
        🕵️ **Investigator/Discriminator**: Evaluates image quality
        
        ⏰ **Temporal Model**: Handles time-series data
        """)
        
        st.header("🌤️ Weather Types")
        for key, value in WEATHER_TYPES.items():
            st.markdown(f"**{key}**: {value}")
        
        st.header("🚀 How It Works")
        st.markdown("""
        1. **Upload** your weather-affected satellite image
        2. **Classify** weather condition (auto or manual)
        3. **Generate** clear version using AI
        4. **Compare** before/after results
        5. **Download** the enhanced image
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        - Works best with **256x256** satellite images
        - Try both **auto-detect** and **manual** weather selection
        - **Quality scores** closer to 1.0 are better
        - **Download** enhanced images for further use
        """)
        
        
        if models:
            st.header("🔧 Model Status")
            for name, model in models.items():
                st.success(f"✅ {name.title()}")

if __name__ == "__main__":
    main()