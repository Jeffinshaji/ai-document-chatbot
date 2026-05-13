
# ********HNSW for large dataset use this code ***********
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(vectorstore):
    # 1. Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 2. Convert the HNSW vector store to a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Define the prompt template
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Construct RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



