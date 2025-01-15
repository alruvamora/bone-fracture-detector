import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model('model_fractures.h5')

    
# App configuration
st.markdown(
    '<h1 style="text-align:center;">Bone Fracture Detector</h1>',
    unsafe_allow_html=True
)

def new_analysis():
       # Preprocess image
    def preprocess_image(img):
        # Convert to RGB if the image is grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150))  # Resize the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize
        return img_array

    
    uploaded_file = st.file_uploader('Upload an X-ray image', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Open the image from the uploaded file
        img = Image.open(uploaded_file)
        
        # Preprocess the image
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        
        # Display the prediction above the image
        if prediction[0][0] > 0.5:
            st.markdown(
                '<div style="text-align:center;">'
                '<p style="font-size:40px; color:green; font-weight:bold;">No Fracture</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="text-align:center;">'
                '<p style="font-size:40px; color:red; font-weight:bold;">Fractured</p>'
                '</div>',
                unsafe_allow_html=True
            )
        
        # Display the image
        st.image(img, caption='Uploaded Image', use_column_width=True)

def about():
    
    # Markdown content explaining the application
    markdown_content = """
    
    The **Bone Fracture Detector** is a web-based application that analyzes X-ray images of bones to predict whether the bone is fractured or not. This project leverages a pre-trained machine learning model built with TensorFlow and Keras, and provides an intuitive interface for users via Streamlit.
    
    ---
    
    ## Features
    
    - **Upload and Analyze**: Users can upload X-ray images in `JPG`, `JPEG`, or `PNG` formats.
    - **Real-Time Predictions**: The application processes the uploaded image and predicts whether the bone is:
      - **Fractured**: Highlighted in bold red text.
      - **Not Fractured**: Highlighted in bold green text.
    - **Interactive Display**: The uploaded image is displayed alongside the prediction for easy reference.
    
    ---
    
    ## Installation
    
    To set up and run the application locally, follow these steps:
    
    ### Prerequisites
    Ensure you have the following installed:
    - Python 3.8 or later
    - `pip` package manager
    
    ### Dependencies
    Install the required Python packages:
    ```bash
    pip install tensorflow streamlit pillow numpy
    ```
    ### Model File
    Ensure the pre-trained model file (`model_fractures.h5`) is available in the same directory as the script.
    
    ---
    
    ## Usage
    
    1. Clone this repository:
       ```bash
       git clone https://github.com/alruvamora/bone-fracture-detector.git
       cd bone-fracture-detector
       ```
    
    2. Run the application:
       ```bash
       streamlit run main.py
       ```
    
    3. Open your browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).
    
    4. Upload an X-ray image to get the prediction.
    
    ---
    
    ## File Structure
    
    - `main.py`: The main application script.
    - `model_fractures.h5`: Pre-trained model file for predictions.
    - `README.md`: Documentation file.
    
    ---
    
    ## How It Works
    
    1. **Image Preprocessing**:
       - Converts grayscale images to RGB (if needed).
       - Resizes images to `150x150` pixels.
       - Normalizes pixel values to the range `[0, 1]`.
    
    2. **Prediction**:
       - The preprocessed image is passed to the model for prediction.
       - The model outputs a probability indicating the likelihood of the bone being not fractured.
    
    3. **Result Display**:
       - Predictions are displayed with clear visual indicators for `Fractured` and `Not Fractured`.
    
    ---
    
    ## Future Enhancements
    
    - Improve model accuracy with additional training data.
    - Add support for additional languages.
    - Implement a feature to save prediction results.
    
    ---
      
    """
    
    # Display the markdown content in the Streamlit app
    st.markdown(markdown_content)

def contact():
    
    # Markdown content explaining the application
    markdown_content_info = """

    ## Contact info:
    - GitHub: https://github.com/alruvamora
    - Linkedin: https://www.linkedin.com/in/%C3%A1lvaro-ruedas-29379a180/
    - Mail: alruvamora@gmail.com


    ## About me: 
    Álvaro Ruedas Mora
    
    - Data Scientist from **Ironhack**
    - Industrial and Automation Electronics Engineer from **Polytechnic University of Madrid**
    - MBA from **EAE Business School**
      
    """
    
    # Display the markdown content in the Streamlit app
    st.markdown(markdown_content_info)


# tabs
tabs = st.tabs(["Data Input", "About", "Contact"])

with tabs[0]:
    new_analysis()
with tabs[1]:
    about()
with tabs[2]:
    contact()

