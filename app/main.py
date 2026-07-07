import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.db.database import init_db
from app.verifier.worker import run_worker

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    worker_task = asyncio.create_task(run_worker())
    yield
    worker_task.cancel()


app = FastAPI(title="LLM Cost Autopilot", lifespan=lifespan)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
