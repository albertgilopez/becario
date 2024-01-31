"""Question answering app in Streamlit.

Originally based on this template:
https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py

Run locally as follows:
> PYTHONPATH=. streamlit run question_answering/app.py

Alternatively, you can deploy this on the Streamlit Community Cloud
or on Hugging Face Spaces. For Streamlit Community Cloud do this:
1. Create a github repo
2. Go to Streamlit Community Cloud, click on "New app" and select the new repo
3. Click "Deploy!"
"""

import streamlit as st

# https://python.langchain.com/docs/integrations/callbacks/streamlit
from langchain_community.callbacks import StreamlitCallbackHandler

from agent import load_agent
from utils import MEMORY

st.set_page_config(page_title="M.IA", page_icon="📈")
st.header("Asistente de Marketing (becario)")

strategy_values = {
    "Pedirle cosas": "plan-and-solve",
    "Intentar que solucione un problema": "plan-and-solve",
    "Voy a tener suerte": "zero-shot-react"
}

strategy = st.radio(
    "¿Qué quieres hacer hoy?",
    list(strategy_values.keys())
)

tool_names_values = {
    "Búsqueda en Google": "google-search",
    "Búsqueda en Internet": "ddg-search",
    "Operaciones Matemáticas": "wolfram-alpha",
    "ArXiv": "arxiv",
    "Wikipedia": "wikipedia",
    "Python REPL": "python_repl",
    "Matemáticas PAL": "pal-math",
    "Matemáticas LLM": "llm-math"
}


tool_names = st.multiselect(
    '¿Qué herramientas quieres usar?',
    list(tool_names_values.keys()),
    ["Búsqueda en Internet", "Operaciones Matemáticas", "Wikipedia"]
)

# Convertir los nombres de herramientas seleccionados a sus códigos correspondientes
selected_tool_codes = [tool_names_values[name] for name in tool_names]


if st.sidebar.button("Eliminar el historial de mensajes"):
    MEMORY.chat_memory.clear()

avatars = {"human": "user", "ai": "assistant"}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assert strategy is not None
agent_chain = load_agent(tool_names=selected_tool_codes, strategy=strategy)

assistant = st.chat_message("assistant")
if prompt := st.chat_input(placeholder="Escribe aquí tu pregunta"):
    st.chat_message("user").write(prompt)
    stream_handler = StreamlitCallbackHandler(assistant)
    with st.chat_message("assistant"):
        response = agent_chain.run({
            "input": prompt,
            "chat_history": MEMORY.chat_memory.messages
        }, callbacks=[stream_handler]
        )
        st.write(response)
