import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    ML_PORT = int(os.getenv('ML_PORT', 5001))
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models')
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
