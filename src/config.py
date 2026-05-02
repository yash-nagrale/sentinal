import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 4))

# Ollama Models
OLLAMA_MODELS = ["qwen2.5:3b", "codellama:latest"]
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHECK_URL = "http://localhost:11434/"

# PS1: Foot Wound Config
FOOT_WOUND_LABELS = [
    "a medical photograph of a foot wound or diabetic foot ulcer",
    "a photograph of a healthy foot with no wound",
    "a photograph of an animal or pet",
    "a photograph of food or a meal",
    "a photograph of a landscape, building, or scenery",
    "a photograph of a person's face or portrait",
    "a screenshot, diagram, or document",
    "a random photograph not related to foot wounds",
]
FOOT_WOUND_THRESHOLD = 0.25

# Recommender Config
OSM_HEADERS = {"User-Agent": "SentinAl/1.0 (medical-ai-app)"}
OSM_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
