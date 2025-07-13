import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()  # Load /app/.env
    config = {
        "DATABASE_URL": os.getenv("DATABASE_URL") or "sqlite:///db.sqlite",
        "SECRET_KEY": os.getenv("SECRET_KEY") or "fallback-secret-key-please-change",
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    }
    # Validate (secure best practice)
    required_keys = ["SECRET_KEY"]
    for key in required_keys:
        if not config[key] or config[key] == "fallback-secret-key-please-change":
            raise ValueError(f"{key} not set properly in .env")
    # Debug print
    print("Config loaded:", {k: v for k, v in config.items() if k != "SECRET_KEY"})  # Mask secret
    return config