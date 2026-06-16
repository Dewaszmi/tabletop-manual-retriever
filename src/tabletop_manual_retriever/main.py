from fastapi import FastAPI

from tabletop_manual_retriever.ingest.router import router as ingest_router
from tabletop_manual_retriever.upload.router import router as upload_router

app = FastAPI()
app.include_router(ingest_router)
app.include_router(upload_router)
