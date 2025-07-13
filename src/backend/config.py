from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

def load_config():
    # Robust project root calculation (works from any subdir or Docker)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # backend -> src -> root
    env_path = os.path.join(project_root, '.env')
    if not os.path.exists(env_path):
        logger.error(f".env not found at {env_path}. Check path or create .env in project root.")
        raise FileNotFoundError(f".env not found at {env_path}")
    load_dotenv(env_path)
    logger.info(f"Loaded .env from: {env_path}")
    return env_path