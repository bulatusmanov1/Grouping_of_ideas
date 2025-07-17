from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.embedding import match_new_idea_to_old_db
from db import Company_DB
from db_config import DB_SETTINGS
from pydantic import BaseModel


db = Company_DB(**DB_SETTINGS)


class Idea(BaseModel):
    title: str
    description: str
    idea_id: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="pages"), name="static")
templates = Jinja2Templates(directory="pages")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/add_idea")
async def add_idea(idea: Idea):
    if idea.idea_id:
        try:
            db.add_new_ideas([(idea.idea_id, idea.title, idea.description)])
            db.process_clusters()
            return {"status": "ok", "idea_id": idea.idea_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "ignored", "reason": "idea_id not provided"}



@app.post("/results")
async def get_results(title: str = Form(...), description: str = Form(...)):
    try:
        combined_text = f"{title} {description}"
        results, best_group = match_new_idea_to_old_db(combined_text, db)
        return {
            "results": results,
            "best_group": best_group
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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