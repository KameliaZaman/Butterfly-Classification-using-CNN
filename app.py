import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model_path = 'C:/Users/kamel/Documents/Image Classification/model_checkpoint_manual_effnet.h5'
model = load_model(model_path)

# Define a function to preprocess the input image
def preprocess_image(img):
    # Check if img is a file path or an image object
    if isinstance(img, str):
        # Load and preprocess the image
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    elif isinstance(img, np.ndarray):
        # If img is already an image array, resize it
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    else:
        raise ValueError("Unsupported input type. Please provide a file path or a NumPy array.")

    return img

# Define the classification function
def classify_image(img):
    # Preprocess the image
    img = preprocess_image(img)
    
    # Make predictions
    predictions = model.predict(img)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    
    return f"Predicted Class: {predicted_class}"

# Create a Gradio interface
iface = gr.Interface(fn=classify_image, 
                     inputs="image",
                     outputs="text",
                     live=True)

# Launch the Gradio app
iface.launch()


# In[ ]:




