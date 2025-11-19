import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from backend.db_manager import create_vectorstore_from_text, get_vectorstore
from backend.main import get_response
from backend.config import settings
import hashlib

st.set_page_config(page_title="TextGPT", layout="wide")
st.title("Chat with Text File")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "processed_files_content" not in st.session_state:
    st.session_state.processed_files_content = {}


def get_file_hash(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()


with st.sidebar:
    st.header("Settings")

    if st.session_state.vector_store:
        st.info("Existing database loaded from disk.")
    else:
        st.warning("No database found. Upload and Process a file to start.")

    if st.session_state.processed_files:
        st.subheader("Processed Files:")
        for filename in st.session_state.processed_files.keys():
            st.text(f"✓ {filename}")

    uploaded_files = st.file_uploader(
        "Upload a text file (.txt)", type=["txt"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process File"):
            with st.spinner("Processing file and building vector store..."):
                new_files_count = 0
                duplicate_files = []
                updated_files = []
                all_new_content = []

                for uploaded_file in uploaded_files:
                    text_content = uploaded_file.getvalue().decode("utf-8")
                    file_hash = get_file_hash(text_content)

                    if uploaded_file.name in st.session_state.processed_files:
                        if (
                            st.session_state.processed_files[uploaded_file.name]
                            == file_hash
                        ):
                            duplicate_files.append(uploaded_file.name)
                            continue
                        else:
                            updated_files.append(uploaded_file.name)
                            st.session_state.processed_files[uploaded_file.name] = (
                                file_hash
                            )
                            st.session_state.processed_files_content[
                                uploaded_file.name
                            ] = text_content
                            new_files_count += 1
                    else:
                        st.session_state.processed_files[uploaded_file.name] = file_hash
                        st.session_state.processed_files_content[uploaded_file.name] = (
                            text_content
                        )
                        new_files_count += 1

                if duplicate_files:
                    st.warning(
                        f"Skipped duplicate file(s): {', '.join(duplicate_files)}"
                    )

                if updated_files:
                    st.info(
                        f"Updated file(s) with new content: {', '.join(updated_files)}"
                    )

                if (
                    new_files_count > 0
                    or len(st.session_state.processed_files_content) > 0
                ):
                    all_texts = []
                    for filename in sorted(
                        st.session_state.processed_files_content.keys()
                    ):
                        content = st.session_state.processed_files_content[filename]
                        all_texts.append(f"--- FILE: {filename} ---\n{content}")

                    combined_text = "\n\n".join(all_texts)

                    vector_store = create_vectorstore_from_text(combined_text)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        file_count = len(st.session_state.processed_files)
                        file_names = ", ".join(st.session_state.processed_files.keys())
                        st.session_state.chat_history = [
                            AIMessage(
                                content=f"Hello! {file_count} file(s) have been processed: {file_names}. How can I help you?"
                            ),
                        ]
                        st.success("Vector store created successfully!")
                        st.rerun()

    if st.session_state.processed_files:
        if st.button("Clear All Files"):
            st.session_state.vector_store = None
            st.session_state.processed_files = {}
            st.session_state.processed_files_content = {}
            st.session_state.chat_history = [
                AIMessage(content="Hello! Please upload a text file to start chatting.")
            ]
            st.success("All files cleared!")
            st.rerun()

if "chat_history" not in st.session_state:
    if st.session_state.vector_store:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello! An existing database is loaded. Ask me anything about it."
            )
        ]
    else:
        st.session_state.chat_history = [
            AIMessage(content="Hello! Please upload a text file to start chatting.")
        ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query.strip() != "":
    if not st.session_state.vector_store:
        st.error("Please upload and process a file before asking questions.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        with st.spinner("Thinking..."):
            response = get_response(
                user_query, st.session_state.vector_store, st.session_state.chat_history
            )

        st.session_state.chat_history.append(AIMessage(content=response))
        with st.chat_message("AI"):
            st.write(response)
