import json
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status

app = FastAPI()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("images/" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        await file.close()

    return {"message": f"Successfuly uploaded {file.filename}"}


from pydantic import BaseModel


class ReqType(BaseModel):
    image_name: str


from model import load_model

encoder, decoder, tokenizer = load_model()

from eval import evaluate, plot_attention


@app.post("/predict")
def predict(request: ReqType):
    request_dictionary = request.dict()
    path = str("images/" + request_dictionary["image_name"])
    try:
        result, attention_plot = evaluate(path, encoder, decoder, tokenizer)

        attention_plot = plot_attention(path, result, attention_plot)
        attention_plot = list(attention_plot)
        json_obj = json.dumps({"result": str(result), "canvas": attention_plot})
        return json_obj

    except Exception as e:
        return {"message": f"Unexpected Error {e}"}


@app.get("/version")
async def version():
    return {"version": "1.0"}
