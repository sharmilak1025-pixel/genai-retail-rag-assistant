from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 📁 Load your product catalog (PDF format)
pdf_path = "data/product_catalog.pdf"  # Update path if needed
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ✂️ Chunk the text into manageable pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Number of characters per chunk
    chunk_overlap=50      # Overlap to preserve context
)
chunks = splitter.split_documents(documents)

# 🔍 Preview the first few chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)
