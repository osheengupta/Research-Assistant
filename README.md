# Research Assistant 📈

A powerful research assistant tool that helps analyze multiple news articles simultaneously using artificial intelligence. Powered by Claude 3 AI, this tool allows you to extract insights and ask questions about content using natural language.

![Demo Screenshot](/img/screenshot.png)

## Features

- 📄 **URL Processing**: Ingest and process up to 3 news articles simultaneously
- 💾 **Vector Search**: Milvus-powered semantic search with Hugging Face embeddings
- 🤖 **Advanced NLP**: Utilizes Claude-3 Sonnet for intelligent question answering
- 🎯 **Source Tracking**: Automatically references original article sources
- 🧹 **Persistent Storage**: Maintains processed data between sessions

## Tech Stack

- **Frontend**: Streamlit
- **Language Model**: Claude 3 (Anthropic)
- **Vector Store**: Milvus
- **Embeddings**: Hugging Face (all-MiniLM-L6-v2)
- **Document Processing**: LangChain
- **Environment Management**: python-dotenv

## Prerequisites

- Python 3.8+
- Milvus database running locally or remotely
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/osheens-research-assistant.git
cd osheens-research-assistant
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Package Versions

The project has been tested with the following package versions:
```
anthropic==0.44.0
langchain==0.3.15
langchain-anthropic==0.3.3
langchain-community==0.3.15
langchain-core==0.3.31
langchain-huggingface==0.1.2
langchain-text-splitters==0.3.5
```

## Setting up Milvus

1. Install Milvus using Docker (recommended):
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

2. Verify Milvus is running (default port: 19530)

## Usage

1. Start the application:
```bash
streamlit run app.py
```
2. Enter up to three news article URLs in the sidebar
3. Click "Process URLs" to analyze the articles
4. Ask questions about the content in the main interface
5. Use "Clear Database" to reset when needed

## Features in Detail

### URL Processing
- Supports multiple URLs (up to 3) simultaneously
- Validates URLs before processing
- Shows loading progress with Streamlit spinners

### Text Processing
- Intelligent text splitting with overlap
- Preserves context across chunks
- Efficient document processing pipeline

### Vector Storage
- Uses Milvus for fast similarity search
- Automatically handles document embeddings
- Efficient retrieval for relevant context

### Question Answering
- Powered by Claude 3 AI
- Provides sourced answers
- Maintains context across questions

## Project Structure

```
osheens-research-assistant/
├── app.py              # Main application file
├── .env               # Environment variables
├── requirements.txt   # Project dependencies
├── README.md         # Project documentation
```

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Anthropic's Claude](https://www.anthropic.com/)
- Vector search by [Milvus](https://milvus.io/)
- Document processing by [LangChain](https://www.langchain.com/)
