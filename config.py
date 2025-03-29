import dotenv
import os

dotenv.load_dotenv()

class Config:
    pass
    
    

class AppConfig:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")