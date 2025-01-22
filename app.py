import os
import streamlit as st
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "news_articles"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# UI Setup
st.title("Osheen's Research Assistant 📈")
st.sidebar.title("Configuration")

# URL Input
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_clicked = st.sidebar.button("Process URLs")
clear_clicked = st.sidebar.button("Clear Database")

# Helper function for URL validation
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Main processing logic
if process_clicked:
    valid_urls = [url for url in urls if url and is_valid_url(url)]
    if not valid_urls:
        st.error("Please provide at least one valid URL")
        st.stop()
    
    try:
        with st.spinner('Loading articles...'):
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
        
        with st.spinner('Processing content...'):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=['\n\n', '\n', '. ', '? ', '! ', ' ', '']
            )
            docs = text_splitter.split_documents(data)
            
            st.session_state.vector_store = Milvus.from_documents(
                docs,
                embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name=COLLECTION_NAME,
                drop_old=True
            )
        
        st.success("✅ Articles processed successfully!")

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.stop()

# Q&A Interface
st.header("Ask Questions")
question = st.text_input("Your question:")

if question and st.session_state.vector_store:
    try:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not anthropic_api_key:
            st.error("❌ Missing Anthropic API Key! Add it to your .env file")
            st.stop()

        # Create QA chain
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                max_tokens=1000,
                anthropic_api_key=anthropic_api_key
            ),
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
        
        # Get and display results
        with st.spinner('Analyzing articles...'):
            result = chain.invoke({"question": question})
        
        st.subheader("Answer")
        st.write(result["answer"])
        
        st.subheader("Sources")
        sources = {doc.metadata['source'] for doc in result['source_documents']}
        for source in sources:
            st.write(f"- {source}")

    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")

# Clear database
if clear_clicked:
    from pymilvus import utility
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    st.session_state.vector_store = None
    st.success("✅ Database cleared!")