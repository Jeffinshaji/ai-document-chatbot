# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# def create_qa_chain(vectorstore):
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
#     prompt = ChatPromptTemplate.from_template(template)

#     rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain

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

# ************* for using external knowledge and chat history, use this code ***********

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# def create_qa_chain(vectorstore):

#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     template = """
# You are a helpful AI assistant.

# Follow this order:
# 1. If answer is in chat history → use it
# 2. Else if answer is in context → use it
# 3. Else → use your own knowledge

# Chat History:
# {chat_history}

# Context:
# {context}

# Question:
# {question}

# Answer clearly:
# """
#     prompt = ChatPromptTemplate.from_template(template)

#     def rag_chat(question, chat_history):
#         # get relevant docs
#         docs = retriever.invoke(question)
#         context = "\n".join([doc.page_content for doc in docs])

#         # format prompt
#         formatted_prompt = prompt.format(
#             chat_history="\n".join(chat_history),
#             context=context,
#             question=question
#         )

#         response = llm.invoke(formatted_prompt)
#         return response.content

#     return rag_chat

# ************* for large dataset use this code ***********

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# def create_qa_chain(retriever):
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

#     template = """
#     You are a helpful AI assistant. Use the following context and chat history to answer.
    
#     Chat History: {chat_history}
#     Context: {context}
#     Question: {question}
    
#     Answer:"""
    
#     prompt = ChatPromptTemplate.from_template(template)

#     def rag_chat(question, chat_history):
#         # The retriever now handles the Hybrid Search automatically
#         docs = retriever.invoke(question)
#         context = "\n".join([doc.page_content for doc in docs])

#         formatted_prompt = prompt.format(
#             chat_history="\n".join(chat_history),
#             context=context,
#             question=question
#         )

#         response = llm.invoke(formatted_prompt)
#         return response.content

#     return rag_chat