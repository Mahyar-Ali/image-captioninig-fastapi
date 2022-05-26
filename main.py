from base64 import encode
from enum import Enum
from unittest.mock import patch
from fastapi import FastAPI, HTTPException, status


import uvicorn
from fastapi import File, UploadFile, FastAPI

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("images/"+file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()
        
    return {"message": f"Successfuly uploaded {file.filename}"}


class EssayType(str, Enum):
    image_name = str

from model import load_model
encoder, decoder = load_model()

from eval import evaluate, plot_attention

@app.post("/predict")
def predict(request: EssayType):
    request_dictionary = request.dict()
    path = str("images/"+request_dictionary["image_name"])
    try:
        result,attention_plot = evaluate(path)

        attention_plot = plot_attention(path, result, attention_plot)

        return {"result":result, "attention_plot":list(attention_plot)}

    except:
        return {"message": "Unexpected Error"}

@app.get("/version")
async def version():
    file = open("VERSION", "r")
    return {"version": "1.0"}