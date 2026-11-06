"""Outils de recherche exposés à l'agent LangChain."""
from langchain_core.tools import tool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQuery
from stores import text_index, clip, clip_store

SOURCES: list[dict] = []   # références collectées pendant une question (affichées dans l'UI)


def _fmt(nodes) -> str:
    lines = []
    for n in nodes:
        m = n.metadata
        SOURCES.append(m)
        lines.append(f"[{m.get('source')} p.{m.get('page')} | {m.get('type')}]\n{n.get_content()[:1200]}")
    return "\n\n---\n\n".join(lines) or "Aucun résultat."


def _search(q: str, k: int, typ: str | None):
    kw = {}
    if typ:
        kw["filters"] = MetadataFilters(filters=[MetadataFilter(key="type", value=typ)])
    return [r.node for r in text_index().as_retriever(similarity_top_k=k, **kw).retrieve(q)]


@tool
def search_text(query: str) -> str:
    """Recherche dans le texte du document. À utiliser pour les explications, procédures, définitions."""
    return _fmt(_search(query, 5, "text"))


@tool
def search_tables(query: str) -> str:
    """Recherche dans les tableaux (valeurs, spécifications, résultats chiffrés, comparatifs)."""
    return _fmt(_search(query, 3, "table"))


@tool
def search_figures(query: str) -> str:
    """Recherche de figures/schémas par leur description textuelle (légendes générées)."""
    return _fmt(_search(query, 3, "image"))


@tool
def search_images_clip(query: str) -> str:
    """Recherche visuelle CLIP : trouve les images dont le contenu correspond à la requête (de préférence en anglais)."""
    vec = clip().encode(query).tolist()
    res = clip_store().query(VectorStoreQuery(query_embedding=vec, similarity_top_k=3))
    return _fmt(res.nodes)


TOOLS = [search_text, search_tables, search_figures, search_images_clip]
