"""PDF -> texte / tableaux / images -> embeddings -> PGVector."""
import base64, pathlib
import fitz  # PyMuPDF
import ollama
from PIL import Image
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import config as C
from stores import embed_model, clip, text_store, clip_store


def parse_pdf(pdf_path: str):
    """Retourne (documents_texte, noeuds_tableaux, liste_images)."""
    pdf = pathlib.Path(pdf_path)
    out = pathlib.Path(C.IMG_DIR); out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    texts, tables, images = [], [], []
    for pno, page in enumerate(doc, 1):
        meta = {"source": pdf.name, "page": pno}
        t = page.get_text("text").strip()
        if t:
            texts.append(Document(text=t, metadata={**meta, "type": "text"}))
        for ti, tab in enumerate(page.find_tables().tables):
            md = tab.to_markdown()
            if md.strip():
                tables.append(TextNode(text=md, metadata={**meta, "type": "table", "table_id": ti}))
        # 1) images raster intégrées
        for ii, img in enumerate(page.get_images(full=True)):
            pix = fitz.Pixmap(doc, img[0])
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < 100 or pix.height < 100:
                continue
            p = out / f"{pdf.stem}_p{pno}_img{ii}.png"; pix.save(p)
            images.append({**meta, "image_path": str(p)})
        # 2) figures vectorielles (schémas, ex. architecture du Transformer)
        for di, bbox in enumerate(page.cluster_drawings()):
            if bbox.width > 120 and bbox.height > 120:
                p = out / f"{pdf.stem}_p{pno}_fig{di}.png"
                page.get_pixmap(clip=bbox, dpi=150).save(p)
                images.append({**meta, "image_path": str(p)})
    return texts, tables, images


def _vlm(prompt: str, path: str) -> str:
    r = ollama.chat(
        model=C.VLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [path]}],
        options={"num_ctx": 2048},  # contexte réduit = moins de RAM
    )
    return r["message"]["content"].strip()


def caption(path: str) -> str:
    try:
        return _vlm("Décris précisément cette figure technique (composants, "
                    "flèches, étiquettes, valeurs lisibles) en 3-5 phrases.", path)
    except Exception as e:  # ex. mémoire insuffisante : on n'interrompt pas l'indexation
        print(f"[caption] échec pour {path}: {e}")
        return "Figure sans description automatique."


def ingest_pdf(pdf_path: str, progress=lambda msg: None):
    embed_model()
    progress("Extraction du PDF…")
    texts, tables, images = parse_pdf(pdf_path)

    progress(f"Indexation de {len(texts)} pages et {len(tables)} tableaux…")
    chunks = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(texts)
    ctx = StorageContext.from_defaults(vector_store=text_store())
    VectorStoreIndex(chunks + tables, storage_context=ctx, show_progress=True)

    progress(f"Traitement de {len(images)} images (CLIP + légendes)…")
    model, caption_nodes, clip_nodes = clip(), [], []
    for im in images:
        cap = caption(im["image_path"])
        vec = model.encode(Image.open(im["image_path"]).convert("RGB")).tolist()
        meta = {**im, "type": "image", "caption": cap}
        clip_nodes.append(TextNode(text=cap, embedding=vec, metadata=meta))
        caption_nodes.append(TextNode(text=f"[Figure page {im['page']}] {cap}", metadata=meta))
    if clip_nodes:
        clip_store().add(clip_nodes)
        VectorStoreIndex(caption_nodes, storage_context=ctx)  # légendes interrogeables en texte
    progress("Indexation terminée.")
    return {"pages": len(texts), "tables": len(tables), "images": len(images)}


def describe_photo(path: str) -> str:
    """Décrit une photo fournie par le technicien (ex. panneau de commande)."""
    return _vlm("Décris cette photo d'équipement : voyants (couleur, état), "
                "écrans, étiquettes, codes lisibles. Ne devine pas ce qui est illisible.", path)