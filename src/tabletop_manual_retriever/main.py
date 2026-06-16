from fastapi import FastAPI

from tabletop_manual_retriever.ingest.router import router as ingest_router

app = FastAPI()
app.include_router(ingest_router)
