import tensorflow as tf
import numpy as np
import json
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

# Load model
model = tf.keras.models.load_model("flower_model.keras")

# Load labels
with open("labels.json") as f:
    class_names = json.load(f)

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
async def root():
    return {"status": "API is active. Go to /docs for the UI."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess(image)

    pred = model.predict(img)
    idx_num = np.argmax(pred)
    idx_str = str(idx_num)

    flower_info = class_names.get(idx_str, {
        "common_name": "Unknown",
        "scientific_name": "N/A",
        "genus": "N/A",
        "family": "N/A",
        "taxonomy": "N/A"
    })

    return {
        "common_name": flower_info.get("common_name"),
        "scientific_name": flower_info.get("scientific_name"),
        "family": flower_info.get("family"),
        "genus": flower_info.get("genus"),
        "taxonomy": flower_info.get("taxonomy"),
        "confidence": float(np.max(pred)),
        "index": int(idx_num)
    }