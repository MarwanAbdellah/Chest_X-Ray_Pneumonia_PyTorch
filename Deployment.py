import sys
import os

# Add model paths relative to the current working directory
for model_path in ["Alexnet", "Densenet", "Efficient_net", "Resnet"]:
    model_dir = os.path.join(os.getcwd(), model_path)
    if model_dir not in sys.path:
        sys.path.append(model_dir)

# Import the inference functions from the respective model files
from Inference_alexnet import inference as inference_alexnet 
from Inference_densenet import inference as inference_densenet
from Inference_effinet import inference as inference_efficient_net
from Inference_resnet import inference as inference_resnet

# FastAPI and other imports
from fastapi import FastAPI, HTTPException
import base64
from PIL import Image
import io
from pydantic import BaseModel

app = FastAPI(title="Pneumonia Detection API", version="1.0.0")

class RequestData(BaseModel):
    image: str

async def process_image(req: RequestData) -> Image.Image:
    """Helper function to process the incoming image data"""
    try:
        image_data = io.BytesIO(base64.b64decode(req.image))
        img = Image.open(image_data)
        img.load()
        print(f"Received image: {img}")
        return img
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

async def handle_prediction(img: Image.Image, inference_func) -> dict:
    """Helper function to handle the prediction process"""
    try:
        prediction = inference_func(img)
        print(f"Prediction: {prediction}")
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Prediction failed. Please check the model or input image.")
        
        return {"prediction": prediction}
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_alexnet/")
async def predict_alexnet(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, inference_alexnet)

@app.post("/predict_densenet/")
async def predict_densenet(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, inference_densenet)

@app.post("/predict_efficient_net/")
async def predict_efficient_net(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, inference_efficient_net)

@app.post("/predict_resnet/")
async def predict_resnet(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, inference_resnet)
