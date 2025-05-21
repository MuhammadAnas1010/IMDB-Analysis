# Import necessary libraries
import tensorflow as tf


# Verify TensorFlow version
print(f"TensorFlow version: {tf.__version__}")









assert tf.__version__ == '2.12.0', "Please install TensorFlow 2.12.0 using: pip install tensorflow==2.12.0"

# Load the existing model trained with TensorFlow 2.18.0
try:
    model = tf.keras.models.load_model('gru_model.keras')  # Adjust path if needed, e.g., 'path/to/gru_model.h5'
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise