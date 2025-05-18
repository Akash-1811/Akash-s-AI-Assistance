import streamlit as st
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import os
import tempfile

# Must be the first Streamlit command
st.set_page_config(page_title="Akash's AI Assistance", layout="wide")

# Title
st.title("💬 Akash's AI Assistance")

st.markdown(
    """
    <style>
    /* Make chat message text color black */
    .st-chat-message > div {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("Options")
    mode = st.selectbox("Choose Mode", ["Chat", "Image"])
    
    # Temperature slider
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

    uploaded_file = st.file_uploader("Upload File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if st.button("Clear All"):
        st.session_state.messages = []
        st.session_state.image_outputs = []
        st.experimental_rerun()

# Session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_outputs" not in st.session_state:
    st.session_state.image_outputs = []

# Helper: Load and split document
def load_file(file):
    extension = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    if extension == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(tmp_file_path)
    elif extension == ".txt":
        loader = TextLoader(tmp_file_path)
    else:
        return []

    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

# Load model
llm = Ollama(model="llama3", temperature=temperature)

# Mode: Chat
if mode == "Chat":
    docs = []
    if uploaded_file:
        docs = load_file(uploaded_file)
        if docs:
            embeddings = OllamaEmbeddings(model="llama3")

            vectorstore = FAISS.from_documents(docs, embeddings)
            chain = load_qa_chain(llm, chain_type="stuff")
        else:
            st.warning("Unsupported file or failed to process file.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if uploaded_file and docs:
            relevant_docs = vectorstore.similarity_search(user_input)
            response = chain.run(input_documents=relevant_docs, question=user_input)
        else:
            response = llm(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Mode: Image
elif mode == "Image":
    prompt = st.chat_input("Enter prompt for image generation...")

    if prompt:
        with st.spinner("Generating image..."):
            model_id = "runwayml/stable-diffusion-v1-5"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_id)

            pipe = pipe.to(device)
            image = pipe(prompt).images[0]

            st.session_state.image_outputs.append((prompt, image))

    for prompt_text, img in st.session_state.image_outputs:
        with st.chat_message("user"):
            st.markdown(f"**Prompt:** {prompt_text}")
        with st.chat_message("assistant"):
            st.image(img, caption="Generated Image", use_column_width=True)
