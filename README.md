# Text File Chat Application with RAG

A Streamlit-based application that allows users to chat with their text files using Retrieval-Augmented Generation (RAG) powered by Google's Gemini AI.

## Features

-  Upload and process text files
-  Interactive chat interface
-  Intelligent context-aware responses
-  RAG implementation using LangChain
-  Powered by Google's Gemini AI
-  Persistent vector storage using Chroma DB

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── backend/
│   ├── __init__.py
│   ├── config.py         # Configuration settings
│   ├── db_manager.py     # Vector store management
│   └── main.py          # RAG chain implementation
└── chroma_db/           # Persistent vector storage directory
```

## Prerequisites

- Python 3.10
- Google API Key (Gemini AI)
- Required Python packages (see requirements.txt)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG.git
cd RAG
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# OR
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload a text file using the sidebar interface.

3. Click "Process File" to create the vector store.

4. Start chatting with your document in the main interface!

## Configuration

Key settings in `backend/config.py`:

- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL_NAME`: Google's embedding model (default: "gemini-embedding-001")
- `CHROMA_PERSIST_DIRECTORY`: Location for vector store persistence

## How It Works

1. **Text Processing**: Documents are split into chunks using RecursiveCharacterTextSplitter.

2. **Vector Store**: Text chunks are embedded using Google's Gemini embedding model and stored in a Chroma vector store.

3. **RAG Chain**:
   - Uses conversation history to generate relevant search queries
   - Retrieves context from the vector store
   - Generates responses using Gemini's chat model

## Dependencies

Major dependencies include:
- streamlit
- langchain
- google-ai-generativelanguage
- chromadb
- pydantic-settings

For a complete list, see `requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.