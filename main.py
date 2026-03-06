import os
# Force Keras 3 behavior to handle the 'batch_shape' error we saw
os.environ["TF_USE_LEGACY_KERAS"] = "0" 

import tensorflow as tf
import numpy as np
import json
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

# 1. Setup absolute paths for the cloud environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "flower_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# 2. Load model and labels once during startup with compile=False
try:
    # Adding compile=False skips loading the optimizer/quantization config
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Primary load failed, trying Keras direct: {e}")
    import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH) as f:
    class_names = json.load(f)

def preprocess(img):
    """Resizes and normalizes the image for MobileNet."""
    img = img.resize((224, 224))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
async def root():
    return {"status": "API is active. Go to /docs for the UI."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives image, runs prediction, and returns taxonomy."""
    # Read the file bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run MobileNet prediction
    img = preprocess(image)
    pred = model.predict(img)
    
    idx_num = np.argmax(pred)
    idx_str = str(idx_num)

    # Map index to our detailed 102 flower labels
    flower_info = class_names.get(idx_str, {
        "common_name": "Unknown",
        "scientific_name": "N/A",
        "genus": "N/A",
        "family": "N/A",
        "taxonomy": "N/A"
    })

    # Return standard Python types (float/int) to avoid JSON errors
    return {
        "common_name": flower_info.get("common_name"),
        "scientific_name": flower_info.get("scientific_name"),
        "family": flower_info.get("family"),
        "genus": flower_info.get("genus"),
        "taxonomy": flower_info.get("taxonomy"),
        "confidence": float(np.max(pred)),
        "index": int(idx_num)
    }