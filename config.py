import os
from pydantic import BaseModel, Field
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configuration model using Pydantic
class Config(BaseModel):
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    db_url: str = Field(..., env="DB_URL")

try:
    config = Config(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        db_url=os.getenv("DB_URL")
    )
except ValueError as e:
    raise ValueError(
        "Failed to load configuration. Ensure GEMINI_API_KEY and DB_URL are set in your .env file.\n"
        f"Error details: {e}"
    )

# Initialize database connection pool
DB_POOL = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=config.db_url)

# Test initial connection
try:
    conn = DB_POOL.getconn()
    DB_POOL.putconn(conn)
    print("Database connected successfully!")
except psycopg2.Error as e:
    raise ConnectionError(f"Failed to establish initial database connection: {e}")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=config.gemini_api_key)

# Initialize conversation memory
buffer_memory = ConversationBufferMemory(return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts([""], embeddings)

# State definition
State = dict