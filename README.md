# Poor Man's Perplexity AI

Poor Man's Perplexity AI is a script that harnesses the combined power of multiple search engines and advanced NLP tools to perform searches, summarize content, process PDFs, and interact with GPT-based AI for generating comprehensive summaries and action steps based on the collected data.

## Features

- **Multi-engine Web Search**: Perform web searches using Google and Bing.
- **Content Summarization**: Extract relevant text from webpages and PDFs to produce concise summaries.
- **PDF Processing**: Seamlessly work with PDF documents to extract text.
- **Concurrent Execution**: Process multiple search results simultaneously for faster performance.
- **Advanced NLP Integration**: Utilize NLP tools like NLTK for text processing.
- **AI-Enhanced Outputs**: Interact with OpenAI's GPT models for intelligent summary generation and task outlining.

## Dependencies

This script requires the following Python libraries:

- `requests`
- `beautifulsoup4`
- `google-api-python-client`
- `python-dotenv`
- `PyMuPDF (fitz)`
- `nltk`
- `openai`

Ensure that you have these installed in your Python environment. You can install them using pip:

```bash
pip install requests beautifulsoup4 google-api-python-client python-dotenv PyMuPDF nltk openai
```

Additionally, make sure to download the `punkt` tokenizer models for NLTK:

```python
import nltk
nltk.download('punkt')
```

## Configuration

Before running the script, you will need:

1. Google API and Custom Search Engine keys.
2. Set up your environment variables for the API keys by creating a `.env` file:

```env
API_KEY=your_google_api_key_here
CSE_KEY=your_custom_search_engine_id_here
```

3. OpenAI key setup (make sure to configure it within the script or as an environment variable).

## Usage

Run the script in your terminal. The program runs in an interactive mode.

```bash
python pmp.py
```

When prompted, type your search query into the console. The script will fetch results from both search engines, summarize them, and pass them to the GPT model for further processing.

To exit the program, type `exit`.

## Note

This is a utility script intended for educational and research purposes. It combines various APIs and leverages concurrency for improved performance. The effectiveness of the summaries and AI responses are highly dependent on the provided data quality and the sophistication of the underlying language models.