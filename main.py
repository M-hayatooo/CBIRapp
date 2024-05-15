import io
import json
import os
import tempfile
from typing import List

import database
import ie_cbir_model as model
import image_process
import nibabel as nib
import numpy as np
import requests
import supabase
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import cosine

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_URL = "https://fnoprhmpeqtrnmpsblfu.supabase.co/storage/v1/object/public/mr-images/ie_cbir_model/Imp_RaDOBase++_epo490.pth?t=2024-05-13T07%3A01%3A53.251Z"


def load_model_from_supabase(url):
    # Supabase URLからファイルをダウンロード
    response = requests.get(url)
    if response.status_code == 200:
        buffer = io.BytesIO(response.content)
        net = model.ResNetVAE(32, [[32,1,2],[64,1,2],[128,2,2]])
        net.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        net.eval()
        return net
    else:
        raise Exception(f"Failed to download the file: HTTP {response.status_code}")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/image_recognition")
async def image_recognition(files: List[UploadFile] = File(...)):
    mr_image = io.BytesIO(await files[0].read())
    try:
        # 一時ファイルに保存してNIfTIイメージを読み込む
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(mr_image.getbuffer())
            tmp_file_path = tmp_file.name
        nii_image = nib.load(tmp_file_path)

    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        return {"error": f"File loading failed: {str(e)}"}


    img = nib.as_closest_canonical(nii_image)
    data = img.get_fdata().astype('float32')

    voxel = np.zeros((1, 80, 112, 80))
    voxel[0] = data
    voxel = image_process.preprocess(voxel)
    voxel = voxel.astype(np.float32)

    x = torch.from_numpy(voxel).clone()

    try:
        net = load_model_from_supabase(MODEL_URL)
    except Exception as e:
        print(str(e))

    net.eval()
    with torch.inference_mode():
        mu, logvar, cos_sim, feature_rep = net.encoder(x)
        feature_rep = feature_rep.cpu().detach().numpy()
        input_feature_rep = feature_rep[0]

    input_array = input_feature_rep

    ldrs = database.get_all_ldr()
    ldr_arrays = [(ldr[0], np.array(ldr[1].split(), dtype=float)) for ldr in ldrs]
    similarities = [(ldr[0], 1 - cosine(input_array, ldr[1])) for ldr in ldr_arrays]
    
    top_three_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

    for url, similarity in top_three_similar:
        print(f"臨床情報ファイルへのリンク: {url}, Similarity: {similarity}")

    if top_three_similar:
        return {"urls": [url for url, _ in top_three_similar]}  # 類似したURLをリストで返す
    else:
        return {"error": "No similar images found"}

@app.get("/clinical-info")
async def get_clinical_info(request: Request, urls: str):
    url_list = urls.split(',')
    data_list = []
    for url in url_list:
        response = requests.get(url)
        if response.status_code == 200:
            data_list.append(response.json())
        else:
            data_list.append({"error": f"Failed to load data from {url}"})

    return templates.TemplateResponse("clinical_info.html", {"request": request, "data_list": data_list})


@app.get("/api/fetch_clinical_info")
async def fetch_clinical_info(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTPエラーがあった場合の処理
        data = response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=data)


def find_top_similar(ldr_arrays, input_ldr, top_n=3):
    input_array = np.fromstring(input_ldr.replace(' ', ''), sep=',')
    similarities = [(i, 1 - cosine(input_array, ldr)) for i, ldr in enumerate(ldr_arrays)]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]



@app.post("/cbir")
async def cbir_system():
    return {"message": "Hello World From Fast API"}


@app.get("/mris")
async def get_mris():
    mris = database.get_all_brain_mri()
    return mris



