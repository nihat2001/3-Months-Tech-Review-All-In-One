import os
from dotenv import load_dotenv
import redis.asyncio as redis

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Text, select
from pgvector.sqlalchemy import Vector

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =====================================================================
# 1. ENV & CONFIGURATION LAYER
# =====================================================================
load_dotenv()

# Environment variables setup with sensible fallbacks for development
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Core Database Engine Initialization
engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()

app = FastAPI(title="Universal Hybrid RAG Production-Ready API")
state = {"redis": None}

@app.on_event("startup")
async def startup_event():
    """Initializes external connections and prepares DB schemas dynamically."""
    # Connecting to Redis Async Client
    state["redis"] = redis.from_url(REDIS_URL, decode_responses=True)
    
    # Executing pgvector extension and generating database tables safely
    async with engine.begin() as conn:
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            print("[Warning] Could not auto-create vector extension. Ensure pgvector is enabled on your DB.")
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("shutdown")
async def shutdown_event():
    """Resource cleanup loop to prevent connection pooling memory leaks."""
    if state["redis"]:
        await state["redis"].close()
    await engine.dispose()

async def get_db_session():
    """Dependency injection yield routine supplying transactional sessions."""
    async with AsyncSession(engine) as session:
        yield session

# =====================================================================
# 2. DATA VALIDATION SCHEMAS (Pydantic)
# =====================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The natural language question to ask the AI engine.")

class QueryResponse(BaseModel):
    question: str
    answer: str
    source: str

# =====================================================================
# 3. DATABASE LAYER (SQLAlchemy + pgvector Mapping)
# =====================================================================
class VectorKnowledgeModel(Base):
    """Clean SQLAlchemy model mapping text chunks to vector dimensional matrices."""
    __tablename__ = "vector_knowledge_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    text_content = Column(Text, nullable=False)
    embedding_vector = Column(Vector(1536))  # Custom dimensions mapped to OpenAI standard structures

# =====================================================================
# 4. RETRIEVAL LAYER (No ABC/Abstractmethod Overheads)
# =====================================================================
class BaseRetriever:
    async def retrieve_context(self, query: str, db_session: AsyncSession) -> str:
        return ""

class VectorStoreRetriever(BaseRetriever):
    """Executes vectorized database spatial operations safely."""
    def __init__(self):
        self.embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-small")

    async def retrieve_context(self, query: str, db_session: AsyncSession) -> str:
        try:
            # Transform unstructured string into a mathematical matrix array
            query_vector = await self.embeddings_engine.aembed_query(query)
            
            # Formulating vector spatial distance query routines
            statement = select(VectorKnowledgeModel).order_by(
                VectorKnowledgeModel.embedding_vector.cosine_distance(query_vector)
            ).limit(1)
            
            execution_result = await db_session.execute(statement)
            knowledge_record = execution_result.scalar_one_or_none()
            return knowledge_record.text_content if knowledge_record else ""
        except Exception as e:
            # Prevents structural system crashes if tables are empty or pgvector is missing
            print(f"[Fallback Active] Vector retrieval failed or table empty: {e}")
            return ""

class KnowledgeGraphRetriever(BaseRetriever):
    """Simulates deterministic multi-hop internal knowledge graphs."""
    async def retrieve_context(self, query: str, db_session: AsyncSession) -> str:
        graph_mock_database = {
            "sheki halva": "Sheki Halva -> ORIGINATED_IN -> Sheki City (Manager: Ali Mammadov)",
            "ali mammadov": "Ali Mammadov -> RESIDES_IN -> Baku City && -> CHIEF_EXECUTIVE -> Sheki Halva Factory"
        }
        matched_relationships = []
        for entity_key, structural_relationship in graph_mock_database.items():
            if entity_key in query.lower():
                matched_relationships.append(structural_relationship)
        
        return "\n".join(matched_relationships) if matched_relationships else ""

# =====================================================================
# 5. UNIVERSAL ORCHESTRATION LAYER (LCEL & Redis Integration)
# =====================================================================
@app.post("/api/v1/rag/ask", response_model=QueryResponse)
async def process_rag_pipeline(payload: QueryRequest, db: AsyncSession = Depends(get_db_session)):
    user_query = payload.question
    redis_client = state["redis"]

    # Step A: Cache Lookup Protocol
    try:
        cached_payload = await redis_client.get(user_query) if redis_client else None
        if cached_payload:
            return QueryResponse(question=user_query, answer=cached_payload, source="Redis Cache Layer")
    except Exception as e:
        print(f"[Warning] Redis cache connection failure bypassed: {e}")

    # Step B: Concurrency-Safe Context Aggregation
    vector_engine = VectorStoreRetriever()
    graph_engine = KnowledgeGraphRetriever()
    
    vector_context = await vector_engine.retrieve_context(user_query, db)
    graph_context = await graph_engine.retrieve_context(user_query, db)
    
    # Assembling context data chunks conditionally
    unified_context = ""
    if vector_context:
        unified_context += f"Vector Context:\n{vector_context}\n\n"
    if graph_context:
        unified_context += f"Graph Relations:\n{graph_context}"

    # Step C: LangChain LCEL Routing Block with Dynamic Universal Prompt
    llm_node = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    rag_prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful, universal AI Assistant.
    
    If the 'Context Data' provided below contains useful and relevant facts to answer the question, prioritize using it.
    If the 'Context Data' is empty, missing, or completely irrelevant, ignore it and answer the question accurately using your own internal general knowledge.
    
    Context Data:
    {context}
    
    User Question: {question}
    Response:
    """)

    # Standard LCEL construction chaining inputs safely through a single pipeline
    lcel_orchestrator = (
        {"context": lambda input_data: unified_context, "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm_node
        | StrOutputParser()
    )

    synthesized_answer = await lcel_orchestrator.ainvoke(user_query)

    # Step D: Cache Result for subsequent performance optimization
    try:
        if redis_client:
            await redis_client.setex(user_query, 3600, synthesized_answer)
    except Exception as e:
        print(f"[Warning] Failed to write cache payload to Redis instance: {e}")

    return QueryResponse(
        question=user_query, 
        answer=synthesized_answer, 
        source="Universal Hybrid RAG Core Engine"
    )
