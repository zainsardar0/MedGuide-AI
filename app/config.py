import os 
import certifi 
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    openai_api_key: str = ''
    pinecone_api_key: str = ''
    tavily_api_key: str = ''
    pinecone_index: str = 'medical-agent-demo'
    openai_chat_model: str = 'gpt-4.1-mini'
    openai_embed_model: str = 'text-embedding-3-small'
    openai_vision_model: str = 'gpt-4.1-mini'
    api_token: str = ''
    allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'

    # Retrieval & routing thresholds (previously hardcoded in agent.py / knowledge.py)
    kb_top_k: int = 5
    kb_min_score: float = 0.30
    web_fallback_score: float = 0.62

    # Knowledge-base description, injected into the agent's system prompt
    kb_name: str = 'Gale Encyclopedia of Medicine'
    kb_year: str = '2002'
    kb_coverage: str = 'entries A-B'

    # Deployment hardening: set APP_ENV=production when deploying
    app_env: str = 'development'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')



settings = Settings()