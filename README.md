# 🌸 Flower Identification API

A **FastAPI-based machine learning API** that classifies flowers using a trained deep learning model.

The API takes an **image of a flower** and returns:

* Common Name
* Scientific Name
* Family
* Genus
* Taxonomy
* Model Confidence

The model is trained on the **102 Flower Dataset** using **TensorFlow/Keras**.

---

# 🚀 Features

* Image-based flower classification
* 102 flower categories
* Deep learning model built with TensorFlow
* REST API using FastAPI
* Ready for frontend integration (React / Flutter)

---

# 📁 Project Structure

```
flower_identification_api/
│
├── main.py              # FastAPI application
├── flower_model.keras   # Trained deep learning model
├── labels.json          # Flower metadata
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python runtime version
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Ishan-prog101/flower_identification_api.git
cd flower_identification_api
```

Create virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the API

Start the server using:

```
uvicorn main:app --reload
```

The API will run at:

```
http://127.0.0.1:8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoint

## Predict Flower

**POST** `/predict`

Upload an image file.

Example response:

```
{
  "common_name": "Marigold",
  "scientific_name": "Tagetes",
  "family": "Asteraceae",
  "genus": "Tagetes",
  "taxonomy": "Plantae > Angiosperms > Eudicots",
  "confidence": 0.92,
  "index": 17
}
```

---

# 🧠 Model Details

* Framework: TensorFlow / Keras
* Image size: 224 × 224
* Dataset: Oxford 102 Flower Dataset
* Output classes: 102 flower species

---

# 🌍 Future Improvements

* Deploy API to cloud
* Mobile app integration
* Add 3D campus map for flora identification
* Expand dataset with more species

---

# 👨‍💻 Authors

Developed as part of a **Project Based Learning (PBL)** project.

Author:
**Ishan Upadhyay**

---
