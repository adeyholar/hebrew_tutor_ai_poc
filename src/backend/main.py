from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import os
import logging
from config import load_config  # Modular import

# Load config early (modular, ensures secrets available)
load_config()
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
    if not ssl_cert or not ssl_key:
        logger.error("SSL_CERT_PATH or SSL_KEY_PATH not set in .env. HTTPS disabled.")
        uvicorn.run(app, host="0.0.0.0", port=8000)  # Fallback to HTTP if cert missing
    else:
        logger.info(f"Using cert: {ssl_cert}, key: {ssl_key}")
        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile=ssl_cert, ssl_keyfile=ssl_key)