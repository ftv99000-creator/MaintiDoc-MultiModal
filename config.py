import os

PG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    user="maintidoc", password="maintidoc", database="maintidoc",
)
EMBED_MODEL, EMBED_DIM = "BAAI/bge-m3", 1024          # texte (multilingue, open source)
CLIP_MODEL, CLIP_DIM = "clip-ViT-B-32", 512            # images
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")       # génération + tool calling
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")     # légendes d'images
IMG_DIR = "data/images"