from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import config as C
from tools import TOOLS

SYSTEM = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance.
Règles :
- Réponds UNIQUEMENT à partir des résultats des outils de recherche. Appelle-les avant de répondre.
- Choisis l'outil adapté : texte (search_text), tableaux/valeurs (search_tables),
  schémas (search_figures, search_images_clip). Tu peux en enchaîner plusieurs
  et reformuler la requête si le premier résultat est insuffisant.
- Cite chaque affirmation sous la forme [fichier p.N].
- Si l'information est absente, dis-le clairement ; n'invente rien.
- Tu ne pilotes aucun équipement. Pour toute action à risque, renvoie à un technicien habilité.
- Réponds dans la langue de l'utilisateur."""


def build_agent():
    llm = ChatOllama(model=C.LLM_MODEL, temperature=0)
    return create_agent(llm, TOOLS, system_prompt=SYSTEM)


def ask(agent, history: list[dict], question: str) -> str:
    msgs = history + [{"role": "user", "content": question}]
    out = agent.invoke({"messages": msgs})
    return out["messages"][-1].content
