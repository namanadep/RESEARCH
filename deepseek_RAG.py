import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Constants and Model Initialization
PDF_STORAGE_PATH = '.'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:8b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:8b")
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Custom CSS with improved styling
st.markdown("""
    <style>
    /* Base Theme */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #00FFAA;
        margin-bottom: 2rem;
    }
    
    /* Chat Interface */
    .chat-container {
        background-color: #1A1A1A;
        border-radius: 15px;
        padding: 20px;
        margin: 2rem 0;
    }
    
    .stChatInput {
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .stChatInput input {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 2px solid #00FFAA !important;
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Message Styling */
    .stChatMessage {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #2D2D2D !important;
        border-left: 4px solid #00FFAA !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1E1E1E !important;
        border-left: 4px solid #007ACC !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        background-color: #1E1E1E;
        border: 2px dashed #00FFAA;
        border-radius: 10px;
        padding: 20px;
        margin: 2rem 0;
    }
    
    /* Status Messages */
    .success-message {
        background-color: #1E3D2D;
        color: #00FFAA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #00FFAA !important;
        font-weight: 600;
    }
    
    .stMarkdown {
        color: #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    return PDFPlumberLoader(file_path).load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def process_document(uploaded_file):
    saved_path = save_uploaded_file(uploaded_file)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Main UI
def main():
    st.markdown("## 📘 Your Intelligent Document Assistant")
    st.markdown("<div class='main-header'>Using Deepseek R1 8B model</div>", unsafe_allow_html=True)
    
    with st.container():
        uploaded_pdf = st.file_uploader(
            "Upload Research Document (PDF)",
            type="pdf",
            help="Select a PDF document for analysis",
            accept_multiple_files=False
        )
    
    if uploaded_pdf:
        with st.spinner("Processing document..."):
            process_document(uploaded_pdf)
        st.markdown("<div class='success-message'>✅ Document processed successfully!</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            user_input = st.chat_input("Ask your question about the document...")
            
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.spinner("Analyzing document..."):
                    relevant_docs = find_related_documents(user_input)
                    ai_response = generate_answer(user_input, relevant_docs)
                
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(ai_response)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
