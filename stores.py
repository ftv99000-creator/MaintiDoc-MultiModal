from functools import lru_cache
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sentence_transformers import SentenceTransformer
import config as C


@lru_cache
def embed_model():
    m = HuggingFaceEmbedding(model_name=C.EMBED_MODEL)
    Settings.embed_model = m
    Settings.llm = None  # on n'utilise LlamaIndex que pour l'indexation/retrieval
    return m


@lru_cache
def clip():
    return SentenceTransformer(C.CLIP_MODEL)


def text_store():   # texte + tableaux + légendes d'images  (table: data_text_chunks)
    return PGVectorStore.from_params(**C.PG, table_name="text_chunks", embed_dim=C.EMBED_DIM)


def clip_store():   # vecteurs CLIP des images           (table: data_clip_images)
    return PGVectorStore.from_params(**C.PG, table_name="clip_images", embed_dim=C.CLIP_DIM)


def text_index():
    embed_model()
    return VectorStoreIndex.from_vector_store(text_store())
