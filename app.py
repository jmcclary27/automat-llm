# app.py
import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import weaviate
from weaviate.auth import AuthApiKey

from automat_llm.core import (
    load_json_as_documents,
    load_personality_file,
    init_interactions,
    generate_response,
    create_rag_chain,
)

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="automat-llm")

# Global state, initialized on startup
STATE = {
    "client": None,
    "documents": None,
    "rag_chain": None,
    "personality": None,
    "user_interactions": None,
    "rude_keywords": ["stupid", "idiot", "dumb", "shut up"],  # adjust to your needs
}


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


@app.on_event("startup")
def startup():
    """
    Runs once when FastAPI starts.
    Initializes Weaviate client, loads documents, creates rag_chain, loads personality and interactions.
    """
    weaviate_url = os.environ.get("WEAVIATE_URL") or os.environ.get("WEAVIATE_CLUSTER_URL") or os.environ.get("WCS_URL")
    if not weaviate_url:
        raise RuntimeError(
            "WEAVIATE_URL (or WEAVIATE_CLUSTER_URL / WCS_URL) is not set. "
            "Set it to your Weaviate REST endpoint, like https://xxxx.weaviate.network"
        )

    weaviate_key = _require_env("WEAVIATE_API_KEY")
    _ = _require_env("GROQ_API_KEY")  # just validate early

    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_key),
    )

    # Load personality + interactions
    personality = load_personality_file()
    user_interactions = init_interactions()

    # Load and upload docs, adjust directory as needed
    # Example assumes you have a ./memories directory with JSON files
    data_dir = os.path.join(os.getcwd(), "memories")
    if not os.path.isdir(data_dir):
        logging.warning(f"No memories directory found at {data_dir}. Skipping document load.")
        documents = []
    else:
        documents = load_json_as_documents(client, data_dir)

    # Build the RAG chain
    rag_chain = create_rag_chain(client, user_id="global", documents=documents)

    STATE["client"] = client
    STATE["documents"] = documents
    STATE["rag_chain"] = rag_chain
    STATE["personality"] = personality
    STATE["user_interactions"] = user_interactions

    logging.info("FastAPI startup complete, RAG chain ready.")


@app.on_event("shutdown")
def shutdown():
    client = STATE.get("client")
    try:
        if client is not None:
            client.close()
    except Exception:
        pass


@app.get("/health")
def health():
    ok = STATE["rag_chain"] is not None
    return {"ok": ok}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if STATE["rag_chain"] is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized yet")

    # generate_response is sync, run it in a thread so FastAPI stays responsive
    response = await run_in_threadpool(
        generate_response,
        req.user_id,
        STATE["user_interactions"],
        req.message,
        STATE["rude_keywords"],
        STATE["personality"],
        STATE["rag_chain"],
    )

    return ChatResponse(user_id=req.user_id, response=response)
