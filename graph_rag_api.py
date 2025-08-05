import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain community modules
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.chains import GraphCypherQAChain

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Knowledge Graph RAG API")

# Optional: Allow frontend/local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str

# Initialize once on startup
logger.info("🚀 Starting Knowledge Graph RAG API...")

graph = Neo4jGraph()
llm = Ollama(model='llama3', temperature=0.0)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True
)

logger.info("✅ Graph RAG chain ready to receive questions.")


@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        logger.info(f"💬 Received query: {request.query}")
        result = chain.invoke({"query": request.query})
        return {"answer": result.get("result", "🤔 No result returned.")}
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")
