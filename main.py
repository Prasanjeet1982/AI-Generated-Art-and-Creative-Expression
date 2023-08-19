from fastapi import FastAPI
from app import fastapi_app

app = FastAPI()

app.include_router(fastapi_app.app)
