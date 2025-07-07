from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf



model = tf.keras.models.load_model("../saved-models/inception_model.keras")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour dev, sinon mets ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#   image pipeline
def colored_image_preprocessing_pipeline(image_bytes, size=(224, 224)):
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        return None
    image = image.resize(size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_np_arr = np.array(image).astype('float32') / 255.0
    img_np_arr = np.expand_dims(img_np_arr, axis=0)  # Add batch dimension
    return img_np_arr


@app.post("/model-treatement/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = colored_image_preprocessing_pipeline(contents)

    if img_array is None:
        return {"error": "Invalid image file"}

    prediction = model.predict(img_array)
    prediction = prediction.tolist()  # Convert numpy array to list for JSON response
    if prediction[0][0] > 0.5:
        result = "Yes"
    else:
        result = "No"
    return {"prediction": result}


if __name__ == "__main__":
    print("✅ main.py is being executed directly")
