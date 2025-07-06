# Poor Man's Perplexity AI

Poor Man's Perplexity AI is a script that harnesses the combined power of multiple search engines and advanced NLP tools to perform searches, summarize content, process PDFs, and interact with GPT-based AI for generating comprehensive summaries and action steps based on the collected data.

## Features

- **Multi-engine Web Search**: Perform web searches using Google and Bing.
- **Content Summarization**: Extract relevant text from webpages and PDFs to produce concise summaries.
- **PDF Processing**: Seamlessly work with PDF documents to extract text.
- **Concurrent Execution**: Process multiple search results simultaneously for faster performance.
- **Advanced NLP Integration**: Utilize NLP tools like NLTK for text processing.
- **AI-Enhanced Outputs**: Interact with OpenAI's GPT models for intelligent summary generation and task outlining.


## Quick Start

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install requests beautifulsoup4 google-api-python-client python-dotenv PyMuPDF nltk openai rich
   ```

3. **Download NLTK data:**

   ```python
   import nltk
   nltk.download('punkt_tab')
   ```

4. **Configure environment variables:**

   - Create a `.env` file in the project directory with your API keys:

     ```env
     API_KEY=your_google_api_key_here
     CSE_KEY=your_custom_search_engine_id_here
     OPENAI_API_KEY=your_openai_key_here
     ```

5. **Run the app:**

   ```bash
   python pmp.py
   ```

   When prompted, enter your search query. Type `exit` to quit.


---

## Note

This is a utility script intended for educational and research purposes. It combines various APIs and leverages concurrency for improved performance. The effectiveness of the summaries and AI responses depends on the provided data quality and the sophistication of the underlying language models.