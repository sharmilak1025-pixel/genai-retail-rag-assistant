from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 📁 Load and chunk the product catalog
pdf_path = "data/product_catalog.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 🔗 Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🧠 Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 💾 Save FAISS index locally
vectorstore.save_local("faiss_index")
print("✅ FAISS index saved successfully.")
