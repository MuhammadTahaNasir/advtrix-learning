from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Download models to local cache
SentenceTransformer('all-MiniLM-L6-v2')
pipeline("text2text-generation", model="google/flan-t5-base")
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")