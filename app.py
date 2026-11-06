import pathlib, tempfile
import streamlit as st
import tools
from agent import build_agent, ask
from ingest import ingest_pdf, describe_photo

st.set_page_config(page_title="MaintiDoc", page_icon="🔧")
st.title("🔧 MaintiDoc — RAG multimodal")

@st.cache_resource
def get_agent():
    return build_agent()

with st.sidebar:
    pdf = st.file_uploader("Document PDF", type="pdf")
    if pdf and st.button("Indexer le document"):
        p = pathlib.Path(tempfile.gettempdir()) / pdf.name
        p.write_bytes(pdf.getvalue())
        with st.status("Indexation…") as s:
            stats = ingest_pdf(str(p), progress=lambda m: s.update(label=m))
        st.success(f"{stats['pages']} pages, {stats['tables']} tableaux, {stats['images']} images")
    photo = st.file_uploader("Photo (optionnel)", type=["png", "jpg", "jpeg"])

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Votre question technique…"):
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    full_q = q
    if photo:
        p = pathlib.Path(tempfile.gettempdir()) / photo.name
        p.write_bytes(photo.getvalue())
        full_q += f"\n\n[Description automatique de la photo jointe : {describe_photo(str(p))}]"
        st.sidebar.image(photo)
    tools.SOURCES.clear()
    with st.chat_message("assistant"):
        with st.spinner("Recherche…"):
            ans = ask(get_agent(), st.session_state.chat[:-1], full_q)
        st.markdown(ans)
        with st.expander("Sources consultées"):
            seen = set()
            for s in tools.SOURCES:
                key = (s.get("source"), s.get("page"), s.get("type"), s.get("image_path"))
                if key in seen: continue
                seen.add(key)
                st.caption(f"{s.get('source')} — p.{s.get('page')} ({s.get('type')})")
                if s.get("image_path"):
                    st.image(s["image_path"])
    st.session_state.chat.append({"role": "assistant", "content": ans})