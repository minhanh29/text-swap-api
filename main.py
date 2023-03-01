from typing import Union, List
import cv2
import os
import uuid
from time import time
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from text_editor import TextSwapper, ModelFactory

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./keys/text-replace-366410-b1df0306b203.json"

app = FastAPI()
temp_dir = "./temp"
model_factory = ModelFactory("./weights")

FONT_FILE = "./font_list.txt"
with open(FONT_FILE, "r", encoding="utf-8") as f:
    font_list = f.readlines()
font_list = [f.strip().split("|")[-1] for f in font_list]
font_dir = "./fonts"
font_names = [font.split("/")[-1].replace(".ttf", "") for font in font_list]

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_image_into_numpy_array(data, img_path=""):
    img = Image.open(BytesIO(data))
    return np.array(img)


@app.get("/")
def welcome():
    return {"App Name": "Text Swapping Application"}


@app.get("/ping")
def ping():
    return {"Message": "Your are good to go."}


@app.get("/fonts")
async def get_all_fonts():
    return {
        "font_list": font_names
    }

@app.get("/font/{id}", response_class=FileResponse)
async def get_font(id: int):
    if id < 0 or id >= len(font_list):
        return None

    return FileResponse(os.path.join(font_dir, font_list[id]))


@app.post("/edit")
async def edit_image(file: UploadFile = File(...), perspectivePoints: List[int] = [], translate: str = Form()):
    img_name = f"{uuid.uuid4()}_{int(time())}.png"
    img_path = os.path.join(temp_dir, img_name)

    print("Processing", img_name)
    print("Translate", translate == "true")
    # RGB image
    img = load_image_into_numpy_array(await file.read())
    if img.shape[0] < 20 or img.shape[1] < 20:
        return None

    if img.shape[-1] > 3:
        img = img[:, :, :3]
    elif img.shape[-1] < 3:
        img_layer = img[:, :, :1]
        img = np.concatenate([img_layer, img_layer, img_layer], axis=-1)

    text_swapper = TextSwapper(model_factory, img, img_path=img_path, translate=(translate == "true"), unified=True,
                               translated_paragraph=None, perspective_points=perspectivePoints)

    success, mess = text_swapper.detect_text()

    # no text detected
    if not success:
        os.remove(img_path)
        print(mess)
        return {
            "image": "",
            "info": [],
            "success": False,
            "message": mess,
        }

    img_bytes = text_swapper.extract_background()
    info = text_swapper.get_response()

    # remove temp file
    os.remove(img_path)

    print("Done", img_name)
    return {
        "image": img_bytes,
        "info": info,
        "success": True,
        "message": "Success"
    }


@app.post("/inpaint")
async def inpaint_image(imgFile: UploadFile = File(...), maskFile: UploadFile = File(...), padList: List[int] = []):
    img_name = f"inpaint_img_{uuid.uuid4()}_{int(time())}.png"
    mask_name = img_name.replace("inpaint_img", "mask")

    print("Inpainting", img_name)

    img_path = os.path.join(temp_dir, img_name)
    mask_path = os.path.join(temp_dir, mask_name)

    img = load_image_into_numpy_array(await imgFile.read(), img_path)
    mask = load_image_into_numpy_array(await maskFile.read(), mask_path)

    if img.shape[0] < 20 or img.shape[1] < 20:
        return None

    img = img[:, :, :3]

    # preprocess the mask
    mask[mask > 0] = 255
    mask = mask[:, :, 0]
    target_width = max(img.shape[1] - padList[0] - padList[2], 1)
    target_height = max(img.shape[0] - padList[1] - padList[3], 1)
    mask = cv2.resize(mask, (target_width, target_height))

    # pad the mask
    pad_mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    pad_mask[padList[1]:-padList[3], padList[0]:-padList[2]] = mask

    result = model_factory.extract_background(img, pad_mask)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', result)

    print("Done Inpainting", img_name)
    return Response(content=buffer.tobytes(), media_type="image/png")
