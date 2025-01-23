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

## Contact info:
- GitHub: https://github.com/alruvamora
- Linkedin: https://www.linkedin.com/in/%C3%A1lvaro-ruedas-29379a180/
- Mail: alruvamora@gmail.com


## About me: 
Álvaro Ruedas Mora

- Data Scientist from **Ironhack**
- Industrial and Automation Electronics Engineer from **Polytechnic University of Madrid**
- MBA from **EAE Business School**
