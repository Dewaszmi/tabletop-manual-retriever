from fastapi import FastAPI

from tabletop_manual_retriever.ingest.router import router as ingest_router
from tabletop_manual_retriever.rag.router import router as rag_router
from tabletop_manual_retriever.upload.router import router as upload_router
from tabletop_manual_retriever.web.router import router as web_router

app = FastAPI()
app.include_router(web_router)
app.include_router(ingest_router)
app.include_router(rag_router)
app.include_router(upload_router)
