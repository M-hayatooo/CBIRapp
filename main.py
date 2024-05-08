import database
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World From Fast API"}


@app.post("/cbir")
async def cbir_system():

    return {"message": "Hello World From Fast API"}

@app.get("/mris")
async def get_mris():
    mris = database.get_all_brain_mri()
    return mris
