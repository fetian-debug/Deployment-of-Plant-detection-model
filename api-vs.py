from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("G:\PlantDiseaseDetectionFor14Classes/savedmodel.h5")
CLASS_NAMES = ["Apple_Cedar_apple_rust", "Apple_healthy", "Corn_Common_rust_", "Corn_healthy",
               "Pepper_bell_Bacterial_spot",
               "Pepper_bell_healthy", "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
               "Strawberry_healthy", "Strawberry_Leaf_scorch", "Tomato_Early_blight", "Tomato_Late_blight",
               "Tomato_healthy"]


@app.get("/ping")
async def ping():
    return "Hello iam alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return images


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

