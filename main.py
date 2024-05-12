import io
from typing import List

import database
import ie_cbir_model as model
import image_process
import nibabel as nib
import numpy as np
import supabase
import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import cosine

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/image_recognition")
async def image_recognition(files: List[UploadFile] = File(...)):
    mr_image = io.BytesIO(await files[0].read())
    print(mr_image)
    try:
        nii_image = nib.load(mr_image)
    except Exception as e:
        return {"error": f"ファイルの読み込みに失敗しました: {str(e)}"}
    
    mr_image = nib.load(nii_image)
    img = nib.as_closest_canonical(mr_image)
    data = img.get_fdata().astype('float32')

    print("脳画像を処理しています...")

    voxel = np.zeros((1, 80, 112, 80))
    voxel[0] = data
    voxel = image_process.preprocess(voxel)
    
    voxel_ = voxel.astype(np.float32)
    x = torch.from_numpy(voxel_).clone()

    net = model.ResNetVAE(32, [[32,1,2],[64,1,2],[128,2,2]])
    net.load_state_dict(torch.load("/Imp_RadoBaseplus_epo490.pth"))

    net.eval()
    with torch.inference_mode():
        mu, logvar, cos_sim, feature_rep = net.encoder(x)
        feature_rep = feature_rep.cpu().detach().numpy()
        input_feature_rep = feature_rep[0]
    

    mris = database.get_all_brain_mri()
    ldrs = database.get_all_ldr()
    ldr_arrays = [np.fromstring(ldr[0].replace(' ', ''), sep=',') for ldr in ldrs]

    input_array = input_feature_rep
    similarities = [(i, 1 - cosine(input_array, ldr)) for i, ldr in enumerate(ldr_arrays)]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    return {"message": "脳画像が正常に処理されました。"}

    return similarities[:3]
    feature_rep_losses = []
    for case in mris:
        feature_rep_loss = np.linalg.norm(input_feature_rep - case.featuer_rep)
        feature_rep_losses.append(feature_rep_loss)



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
    # feature_rep_losses = []
    # for case in mris:
    #     feature_rep_losses.append(case.featuer_rep)

    # min_idx = np.argmin(feature_rep_losses)
    # uid = mris[min_idx].img_path
    # subject = f"{uid}"

    # res = supabase.storage.from_('mr-images').get_public_url(f'mr-images/{subject}')

    # return res
    return mris



