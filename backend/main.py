from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from backend.config import settings
 
 
def get_context_retriever_chain(vector_store):
    """
    Creates a retriever chain that is aware of the conversation history.
 
    Args:
        vector_store (Chroma): The vector store containing document embeddings.
 
    Returns:
        RetrievalChain: The history-aware retriever chain.
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  
        google_api_key=settings.GOOGLE_API_KEY,
    )
 
    retriever = vector_store.as_retriever()
 
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
 
 
def get_conversational_rag_chain(retriever_chain):
    """
    Creates a conversational RAG chain for answering questions.
 
    Args:
        retriever_chain (RetrievalChain): The history-aware retriever chain.
 
    Returns:
        RetrievalChain: The conversational RAG chain.
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  
        google_api_key=settings.GOOGLE_API_KEY,
    )
 
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
 
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
 
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
 
 
def get_response(user_input, vector_store, chat_history):
    """
    Gets a response from the conversational RAG chain.
 
    Args:
        user_input (str): The user's question.
        vector_store (Chroma): The vector store for retrieval.
        chat_history (list): The conversation history.
 
    Returns:
        str: The generated answer from the RAG chain.
    """
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
 
    response = conversation_rag_chain.invoke(
        {"chat_history": chat_history, "input": user_input}
    )
 
    return response.get("answer", "Sorry, I could not find an answer.")