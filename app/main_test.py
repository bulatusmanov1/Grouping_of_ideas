from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from app.db_test import *

app = FastAPI()
templates = Jinja2Templates(directory="app//pages")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@app.post("/insert")
async def insert_data():
    db = Company_DB()
    #db.init_db()
    db.insert_data('1234', 'adasddssd')
    print("Идея записана")
    db.close()
    return {"message": "Идея записана"}