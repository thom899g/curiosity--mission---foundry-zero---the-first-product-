"""
Archon Core Configuration - Production Settings
Centralized configuration with environment-based overrides
"""
import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from loguru import logger

class Settings(BaseSettings):
    """Main application settings with validation"""
    
    # Firebase Configuration
    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "archon-core-dev")
    firebase_credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./config/firebase-key.json")
    
    # Model Configuration
    model_name: str = os.getenv("MODEL_NAME", "microsoft/phi-2")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models")
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "2048"))
    
    # API Configuration
    api_version: str = "v1"
    api_title: str = "Archon Core API"
    api_description: str = "AI-native architectural immune system"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
    
    # Stripe Configuration
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    # Redis/Celery Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Compute Tiers (in seconds)
    tier0_timeout: int = 1
    tier1_timeout: int = 10
    tier2_timeout: int = 3600
    
    @property
    def firestore_config(self) -> Dict[str, Any]:
        """Generate Firestore configuration dictionary"""
        return {
            "project": self.firebase_project_id,
            "credentials": self.firebase_credentials_path
        }
    
    def validate_configuration(self) -> bool:
        """Validate critical configuration values"""
        errors = []
        
        if not os.path.exists(self.firebase_credentials_path):
            errors.append(f"Firebase credentials not found at {self.firebase_credentials_path}")
        
        if not self.stripe_secret_key:
            logger.warning("Stripe secret key not configured - billing features will be disabled")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info(f"Configuration validated successfully for project {self.firebase_project_id}")
        return True

# Global settings instance
settings = Settings()