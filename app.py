import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# 📦 Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("genai-retail-rag-assistant/faiss_index", embedding_model)

# 🧠 Load Mistral-7B
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# 🔗 Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 🎨 Streamlit UI
st.set_page_config(page_title="Retail RAG Assistant", layout="centered")
st.title("🛍️ Retail Product Query Assistant")

query = st.text_input("Ask a product-related question:")
if query:
    with st.spinner("Generating answer..."):
        response = rag_chain.run(query)
    st.success("Answer:")
    st.write(response)
