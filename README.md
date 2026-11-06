# MaintiDoc — Étape 1 (RAG agentique multimodal)

```bash
docker compose up -d                 # PostgreSQL + pgvector
ollama pull qwen2.5:7b && ollama pull qwen2.5vl:7b
pip install -r requirements.txt
streamlit run app.py
```
Charger le PDF dans la barre latérale -> « Indexer » -> poser des questions
(ex. « Quelle est la valeur de h dans le modèle de base ? », « Montre le schéma de l'architecture »).

Fichiers : `ingest.py` (extraction + indexation), `stores.py` (PGVector, embeddings, CLIP),
`tools.py` (outils de recherche), `agent.py` (agent LangChain), `app.py` (Streamlit).
