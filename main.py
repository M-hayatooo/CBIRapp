import io
from typing import List

import database
import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.templating import Jinja2Templates

import ie_cbir_model as model
import image_process

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/image_recognition")
async def image_recognition(files: List[UploadFile] = File(...)):
    # ここから脳画像の処理
    input_image = files[0].file.read()
    mr_img = (
        nib.squeeze_image(nib.as_closest_canonical(input_image))
        .get_fdata()
        .astype("float32")
    )
    voxel = np.zeros((1, 80, 112, 80))
    voxel[0] = mr_img
    voxel = image_process.preprocess(voxel)
    
    voxel_ = voxel.astype(np.float32)
    x = torch.from_numpy(voxel_).clone()

    net = model.ResNetVAE(32, [[32,1,2],[64,1,2],[128,2,2]])
    net.load_state_dict(torch.load("/Imp_RadoBaseplus_epo490.pth"))

    net.eval()
    with torch.no_grad():
        mu, logvar, cos_sim, feature_rep = net.encoder(x)
        feature_rep = feature_rep.cpu().detach().numpy()
        input_feature_rep = feature_rep[0]
    

    mris = database.get_all_brain_mri()
    feature_rep_losses = []
    for case in mris:
        feature_rep_loss = np.linalg.norm(input_feature_rep - case.featuer_rep)
        feature_rep_losses.append(feature_rep_loss)

        
    return feature_rep_losses



@app.post("/cbir")
async def cbir_system():
    return {"message": "Hello World From Fast API"}


@app.get("/mris")
async def get_mris():
    mris = database.get_all_brain_mri()
    feature_rep_losses = []
    for case in mris:

        feature_rep_losses.append(case.featuer_rep)


    return mris
