from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from dotenv import load_dotenv
import os
import logging

# Load .env file (modular, absolute path for portability)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hebrew Tutor AI PoC", description="Secure API for Tanach Learning with Lexicon and Audio", version="1.0.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@app.get("/health", description="Health check endpoint")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    ssl_cert = os.getenv("SSL_CERT_PATH")
    ssl_key = os.getenv("SSL_KEY_PATH")
    logger.info(f"Using cert: {ssl_cert}, key: {ssl_key}")
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)