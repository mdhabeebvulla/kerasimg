from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
app = FastAPI()
model = ResNet50(weights='imagenet')
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/")
async def root(request: Request):
    
    return templates.TemplateResponse("index.html", {'request': request,})
@app.post("/scorefile/")
async def create_upload_files(request: Request,file: UploadFile = File(...)):
    if 'image' in file.content_type:
        contents = await file.read()
        filename = 'static/' + file.filename
        with open(filename, 'wb') as f:
            f.write(contents)
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    result = decode_predictions(preds, top=3)[0][0][1]
    return templates.TemplateResponse("predict.html", {"request": request,"result":result,})

