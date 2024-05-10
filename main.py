import database
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/cbir")
async def cbir_system():

    return {"message": "Hello World From Fast API"}

@app.get("/mris")
async def get_mris():
    mris = database.get_all_brain_mri()
    return mris
