from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from embedding import match_new_idea_to_old_db
from db import Company_DB
from db_config import DB_CONFIG

db = Company_DB(**DB_CONFIG)

app = FastAPI()
app.mount("/static", StaticFiles(directory="pages"), name="static")
templates = Jinja2Templates(directory="pages")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check_idea/", response_class=HTMLResponse)
async def check_idea(request: Request, title: str = Form(...), description: str = Form(...)):
    combined_text = title + " " + description
    results, best_group = match_new_idea_to_old_db(combined_text, db)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "results": results, 
        "best_group": best_group,
        "title": title,
        "description": description
    })