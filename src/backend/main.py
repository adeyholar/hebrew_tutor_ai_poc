import os
from fastapi import FastAPI, Body
from .config import load_config  # Dotted relative (works with package)

# Temporary debug prints (remove in production)
print("Current working dir:", os.getcwd())
print("Import path successful for config.")

config = load_config()
app = FastAPI(debug=config["DEBUG"])

@app.get("/")
async def root():
  return {"message": "Backend ready"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/lexicon/{word}")
async def get_lexicon(word: str):
    return {"root": "example_root", "definition": "def", "grammar": "gram"}

@app.post("/feedback")
async def feedback(recognizedText: dict = Body(...)):
    text = recognizedText.get("recognizedText", "")
    return {"improvements": f"Improve based on: {text}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="cert.pem", ssl_keyfile="key.pem")  # Adjust paths for local