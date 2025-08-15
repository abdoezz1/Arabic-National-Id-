from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from Arabic_id_detector import SimplifiedArabicIDDetector
import cv2
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist before mounting
os.makedirs("id_images", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static directory to serve ID images to frontend
app.mount("/id_images", StaticFiles(directory="id_images"), name="id_images")

# Initialize detector
detector = SimplifiedArabicIDDetector(
    main_model_path="D:\\Egabi internship\\National_id_deploy\\models\\best.pt",
    digits_model_path="D:\\Egabi internship\\National_id_deploy\\models\\bestd.pt"
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Save input image with UUID
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    temp_file_path = os.path.join("uploads", input_filename)
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    try:
        # Run pipeline
        results = detector.process_id_image(temp_file_path)

        # Decide which image to return (rotated or cropped)
        image_path = results.get("final_processed_image") or results.get("cropped_path")
        if image_path:
            # Use consistent naming to avoid redundant copies
            new_image_name = f"{file_id}_shown.jpg"
            save_path = os.path.join("id_images", new_image_name)
            shutil.copy(image_path, save_path)
            results["id_image_path"] = f"id_images/{new_image_name}"

        return JSONResponse({
            "national_id": results.get("national_id", ""),
            "first_name": results.get("first_name", ""),
            "last_name": results.get("last_name", ""),
            "address1": results.get("address1", ""),
            "address2": results.get("address2", ""),
            "id_image_path": results.get("id_image_path", "")
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
