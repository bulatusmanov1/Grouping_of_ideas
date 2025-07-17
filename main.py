from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from logger import setup_logger
from utils.embedding import match_new_idea_to_old_db
from db import Company_DB
from db_config import DB_SETTINGS
from pydantic import BaseModel
import time
import traceback

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started.")
    yield
    logger.info("Application stopped.")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="pages"), name="static")
templates = Jinja2Templates(directory="pages")

db = Company_DB(**DB_SETTINGS)

class Idea(BaseModel):
    title: str
    description: str
    idea_id: str

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error("Unhandled exception:")
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

        duration = round(time.time() - start_time, 4)
        logger.info(f"Response: {request.method} {request.url.path} - {response.status_code} in {duration}s")
        return response

app.add_middleware(LoggingMiddleware)

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