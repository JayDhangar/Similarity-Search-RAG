import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()
Pinecone_api=os.getenv("API_Key_Pinecone")

st.set_page_config(page_title="PDF chat bot",layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text=""

@st.cache_resource
def llm_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model= llm_model()
Index_name="pdf-chat"

pc=Pinecone(api_key=Pinecone_api)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if Index_name not in existing_indexes:
    pc.create_index(
        name=Index_name,
        dimension=384,          # required for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(Index_name) 

st.sidebar.title("Upload PDF")
uploaded_pdf=st.sidebar.file_uploader("Choose a PDF file to upload",type=["pdf","text","docx"],accept_multiple_files=True)
def read_file(files):
    full_text=""

    for file in files:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

        elif file_extension == "txt":
            content = file.read().decode("utf-8")
            full_text += content + "\n"

        elif file_extension == "docx":
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            loader = Docx2txtLoader(file.name)
            docs = loader.load()
            for doc in docs:
                full_text += doc.page_content + "\n"
            os.remove(file.name)
    return full_text

if uploaded_pdf:
    st.sidebar.success("Uploaded successfully")
    raw_text=read_file(uploaded_pdf) 
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=splitter.split_text(raw_text)

    vectors_emb=[]
    for i,chunk in enumerate(chunks):
        emb=model.encode(chunk).tolist()
        vectors_emb.append((f"chunk-{i}",emb,{"text":chunk}))

    index.upsert(vectors_emb)
    st.session_state.pdf_text=True

st.title("Chat bot")

querry=st.text_input("Start chatting :")

if st.button("start"):
    q_emb=model.encode(querry).tolist()
    result=index.query(vector=q_emb,top_k=3,include_metadata=True)
    
    context=""
    for match in result["matches"]:
            context += match["metadata"]["text"] + "\n"

    ans=f"Answer from PDF: {context[:600]}"
    
    st.session_state.messages.append(("You",querry))
    st.session_state.messages.append(("Bot",ans))
    
    if querry.strip() == "":
        st.warning("Enter your querry")

st.subheader("Chat History")

for role,msg in st.session_state.messages:
    if role=="You":
        st.markdown(f"You:{msg}")
    else:
        st.markdown(f"PDF BOT:{msg}")
