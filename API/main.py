import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants matching training
IMAGE_SIZE = 256
CLASS_NAMES = ["oily", "normal", "dry"]

# Load model
try:
    MODEL = load_model(r"FYP\Training\vit_model_bs4_lr0.1_Adam.pth")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model loading failed")


def read_file_as_image(data) -> np.ndarray:
    """
    Read and convert image data to numpy array.
    """
    try:
        image = Image.open(BytesIO(data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")


def preprocess_image(image: np.ndarray) -> tf.Tensor:
    """
    Preprocess the image: convert to tensor, resize, normalize.
    """
    try:
        image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert to float32
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize
        image = image / 255.0  # Normalize to [0, 1]
        return tf.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")


@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    if file is None:
        logger.warning("No file provided in the request.")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Step 1: Read the uploaded file
        image_data = await file.read()
        image = read_file_as_image(image_data)
        logger.debug(f"Original image shape: {image.shape}")

        # Step 2: Preprocess the image
        image_tensor = preprocess_image(image)
        logger.debug(f"Image tensor shape: {image_tensor.shape}")

        # Step 3: Predict
        pred_array = MODEL.predict(image_tensor)
        logger.debug(f"Raw predictions: {pred_array}")

        # Step 4: Interpret results
        pred_class_index = np.argmax(pred_array[0])
        predicted_class = CLASS_NAMES[pred_class_index]
        confidence = float(np.max(pred_array[0]))

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                name: float(prob)
                for name, prob in zip(CLASS_NAMES, pred_array[0])
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
