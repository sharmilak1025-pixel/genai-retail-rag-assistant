from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

def load_vectorstore():
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    index_path = "/content/drive/MyDrive/genai-retail-rag-assistant/faiss_index"
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

def build_rag_chain(retriever):
    def truncate_context(inputs):
        context = inputs["context"]
        if isinstance(context, list):
            context = " ".join(doc.page_content for doc in context)
        return {
            "context": context[:3000],  # truncate to ~3000 characters
            "question": inputs["question"]
        }

    prompt = PromptTemplate.from_template(
        "Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
    )

    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)

    rag_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | RunnableLambda(truncate_context)
        | prompt
        | llm
    )

    return rag_chain

# Run the chain
retriever = load_vectorstore()
rag_chain = build_rag_chain(retriever)

query = "What are the benefits of the new organic shampoo variant?"
response = rag_chain.invoke(query)
print(response)