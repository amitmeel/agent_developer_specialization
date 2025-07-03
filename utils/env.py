import os
import sys

from dotenv import load_dotenv
from typing import Optional
from pathlib import Path

# Ensure project root is in Python path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized configuration class for environment variables"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.example.com")
    # SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    
    @classmethod
    def get_env_var(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get any environment variable with optional default"""
        return os.getenv(key, default)
    
    @classmethod
    def validate_required_vars(cls) -> None:
        """Validate that all required environment variables are set"""
        required_vars = ["GEMINI_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Create a global config instance
config = Config()
