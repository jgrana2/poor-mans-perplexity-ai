import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import fitz 
import nltk
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import datetime
import math
import re
from collections import Counter

# Function to improve the search query using an LLM
def improve_query_with_llm(user_input):
    client = OpenAI()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prompt = (
        "Given the following user input, generate a highly specific and effective web search query "
        "using the most relevant keywords, context, and details. If the input is vague, infer likely intent and add clarifying terms. "
        "Include relevant entities, locations, dates, or technical terms if present. "
        "Focus on finding current information, news, and factual content rather than definitions or explanations. "
        "Avoid generic terms that might return dictionary definitions or unrelated content. "
        "Only output the improved query, nothing else.\n\n"
        f"Current date and time: {now}\n"
        f"User input: {user_input}"
    )
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": "You are an expert at crafting effective search queries that prioritize current information, news, and factual content over dictionary definitions."},
            {"role": "user", "content": prompt}
        ]
    )
    # For OpenAI v1 API, the improved query is in response.choices[0].message.content
    return response.choices[0].message.content.strip()

# Function to perform a Google search
def google_search(query, num_results=10, recency_days=None):
    try:
        api_key = os.getenv('API_KEY')
        cse_key = os.getenv('CSE_KEY')
        # Build the resource object for Custom Search
        resource = build("customsearch", 'v1', developerKey=api_key).cse()

        params = {"q": query, "cx": cse_key, "num": num_results}
        result = resource.list(**params).execute()

        # Ensure the result is a dictionary and has the 'items' key
        if isinstance(result, dict) and 'items' in result:
            return result['items']
        else:
            print('The result does not contain items key')
            return []

    except Exception as e:
        # If an exception occurs, print the error and return an empty list
        print(f"An error occurred during the Google search: {e}")
        return []

# Function to perform a Bing search
def bing_search(query, num=10, recency_seconds=None):
    headers = get_headers()
    qft_param = f"&qft=+filterui:age-lt{recency_seconds}" if recency_seconds else ""
    bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}{qft_param}"
    response = safe_request(bing_url, headers)
    if not response:
        return []
    return parse_bing_response(response, num)

# Helper Functions
def get_headers():
    return {'User-Agent': 'Mozilla/5.0'}

def safe_request(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises HTTPError if status_code is not 200
        return response
    except requests.Timeout:
        print(f"[ERROR] Timeout fetching {url}")
        return None
    except requests.RequestException as e:
        print(f"Request to {url} failed: {e}")
        return None

def extract_real_url(bing_url):
    """Extract the actual destination URL from Bing redirect URLs"""
    if 'bing.com/ck/a?' in bing_url:
        try:
            # Parse the URL to extract the 'u' parameter
            parsed = urllib.parse.urlparse(bing_url)
            params = urllib.parse.parse_qs(parsed.query)
            if 'u' in params:
                # The 'u' parameter contains the encoded destination URL
                encoded_url = params['u'][0]
                # Decode the URL
                real_url = urllib.parse.unquote(encoded_url)
                return real_url
        except Exception as e:
            print(f"[DEBUG] Failed to extract real URL from {bing_url}: {e}")
    return bing_url  # Return original if extraction fails

def parse_bing_response(response, num):
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    # Dictionary/definition domains to exclude
    exclude_domains = [
        'dle.rae.es', 'definiciones-de.com', 'thefreedictionary.com', 
        'buscapalabra.com', 'palabras.help', 'bab.la', 'wiktionary.org',
        'wordreference.com', 'glosbe.com', 'ganador.co.jp'
    ]
    
    for result in soup.find_all('li', {'class': 'b_algo'}, limit=num*2):  # Get more results to filter
        title_element = result.find('h2')
        link_element = result.find('a')
        if title_element and link_element:
            title = title_element.text.lower()
            original_link = link_element['href']
            
            # Extract real URL from Bing redirect
            real_link = extract_real_url(original_link)
            
            # Skip dictionary/definition results
            if any(domain in real_link for domain in exclude_domains):
                continue
            if any(word in title for word in ['definición', 'definition', 'diccionario', 'dictionary']):
                continue
                
            results.append({'title': title_element.text, 'link': real_link})
            if len(results) >= num:
                break
    return results

# Add other functions for summarization, PDF processing, etc.
import io

def read_pdf_content(response):
    pdf_bytes = response.content  # Get the PDF as bytes
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def is_pdf_response(response):
    content_type = response.headers.get('Content-Type', '')
    content_disposition = response.headers.get('Content-Disposition', '')

    return ('application/pdf' in content_type.lower() or
            'attachment' in content_disposition.lower())

def read_link_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.Timeout:
        print(f"[ERROR] Timeout fetching {url}")
        return {'title': "Timeout", 'text': "", 'main_image': None}
    except requests.HTTPError as e:
        print(f"[ERROR] HTTP error fetching {url}: {e}")
        return {'title': "HTTP Error", 'text': "", 'main_image': None}
    except Exception as e:
        print(f"[ERROR] Exception fetching {url}: {e}")
        return {'title': "Error", 'text': "", 'main_image': None}

    # Check if response has content
    if not response.content:
        print(f"[WARN] Empty response from {url}")
        return {'title': "Empty Response", 'text': "", 'main_image': None}
    
    if url.endswith('.pdf') or is_pdf_response(response):
        # The content is a PDF file
        text = read_pdf_content(response)
        title = "PDF Document"
        main_image = "No main image"
    else:
        # The content is HTML
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"[ERROR] Failed to parse HTML from {url}: {e}")
            return {'title': "Parse Error", 'text': "", 'main_image': None}
            
        # Extract the title
        title = soup.title.string if soup.title else "No title"
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from multiple content elements
        content_elements = soup.find_all(['p', 'div', 'span', 'article', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        text_parts = []
        for element in content_elements:
            element_text = element.get_text(strip=True)
            if element_text and len(element_text) > 10:  # Only include substantial text
                text_parts.append(element_text)
        
        text = ' '.join(text_parts)
        
        if not text.strip():
            print(f"[WARN] No text content extracted from {url}")
        
        # Extract the main image
        main_image = soup.find('img')['src'] if soup.find('img') else None

    return {'title': title, 'text': text, 'main_image': main_image}

def preprocess_text(text):
    """Clean and preprocess text for better analysis"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    return text.strip()

def calculate_tf_idf(sentences, query_terms=None):
    """Calculate TF-IDF scores for sentences"""
    # Preprocess sentences and create word lists
    processed_sentences = []
    for sentence in sentences:
        # Convert to lowercase and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in words if len(word) > 2 and word not in stop_words]
        processed_sentences.append(words)
    
    # Calculate document frequency for each word
    doc_freq = Counter()
    for words in processed_sentences:
        unique_words = set(words)
        for word in unique_words:
            doc_freq[word] += 1
    
    # Calculate TF-IDF scores for each sentence
    sentence_scores = []
    total_docs = len(processed_sentences)
    
    for i, words in enumerate(processed_sentences):
        if not words:  # Skip empty sentences
            sentence_scores.append(0)
            continue
            
        # Calculate term frequency for this sentence
        term_freq = Counter(words)
        
        # Calculate TF-IDF score
        tfidf_score = 0
        for word, tf in term_freq.items():
            # TF: term frequency in document
            tf_normalized = tf / len(words)
            # IDF: inverse document frequency
            idf = math.log(total_docs / (doc_freq[word] + 1))
            tfidf_score += tf_normalized * idf
        
        # Boost score if sentence contains query terms
        if query_terms:
            query_boost = 0
            sentence_lower = sentences[i].lower()
            for term in query_terms:
                if term.lower() in sentence_lower:
                    query_boost += 1
            tfidf_score *= (1 + query_boost * 0.5)  # 50% boost per query term
        
        sentence_scores.append(tfidf_score)
    
    return sentence_scores

def summarize_text(text, n, query_terms=None):
    """Improved summarization using TF-IDF scoring"""
    if not text or len(text) < 100:
        return text
    
    # Preprocess the text
    text = preprocess_text(text)
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Filter out very short sentences or those with too many non-alphabetic characters
    filtered_sentences = []
    for sentence in sentences:
        if len(sentence) > 20 and len(re.findall(r'[a-zA-Z]', sentence)) > len(sentence) * 0.5:
            filtered_sentences.append(sentence)
    
    if not filtered_sentences:
        return text[:500]  # Return first 500 chars if no good sentences found
    
    # Calculate TF-IDF scores
    sentence_scores = calculate_tf_idf(filtered_sentences, query_terms)
    
    # Adjust summary length based on content - keep it small for cost efficiency
    content_length = len(filtered_sentences)
    if content_length < 3:
        n = min(n, content_length)
    elif content_length < 8:
        n = min(n, 2)
    else:
        n = min(n, 3)  # Maximum 3 sentences per summary
    
    # Sort sentences by score and select top n
    sentence_score_pairs = list(zip(filtered_sentences, sentence_scores))
    sentence_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top sentences and sort by original order
    top_sentences = [pair[0] for pair in sentence_score_pairs[:n]]
    
    # Sort by original order in text
    ordered_summary = []
    for sentence in filtered_sentences:
        if sentence in top_sentences:
            ordered_summary.append(sentence)
    
    return " ".join(ordered_summary)

def fetch_and_summarize(result, query_terms=None):
    try:
        content = read_link_content(result['link'])
        text_length = len(content["text"])
        print(f"[DEBUG] Extracted {text_length} characters from {result['link']}")
        
        if text_length < 50:
            print(f"[WARN] Very little content extracted from {result['link']}")
            return ""
            
        summarized_content = summarize_text(content["text"], 3, query_terms)
        print(f"[DEBUG] Summary length: {len(summarized_content)} characters")
        return summarized_content
    except Exception as exc:
        print(f"[ERROR] An error occurred for {result['link']}: {exc}")
        return ""

# Function to print the title and link, and return a summarized version
def fetch_print_summarize(item, query_terms=None):
    try:
        title = item.get('title')
        link = item.get('link')
        print(f"[INFO] Fetching: {link}")
        # Fetch and summarize the content of the link
        summary = fetch_and_summarize({'link': link}, query_terms)
        print(f"[INFO] Done: {link}")
        return f"{title}, {link}\n{summary}\n"
    except Exception as exc:
        print(f'[ERROR] URL resulted in an exception: {exc}')
        return ""

def sendToGPT(copyPrompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": copyPrompt}
        ],
        stream=True
    )
    console = Console()
    full_response = ""
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                live.update(Markdown(full_response))
    # Optionally print the final markdown after streaming
    # console.print(Markdown(full_response))

def sendToOllama(copyPrompt):
    import ollama
    stream = ollama.chat(model='mistral', messages=[
        {
            'role': 'user',
            'content': copyPrompt,
        },
        ], stream=True,)
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
  
# Example usage
if __name__ == "__main__":
    load_dotenv()
    nltk.download('punkt_tab')

    while True:
        query = input("Please enter the search query: ")
        if query.lower() == "exit":
            print("Exiting the program.")
            break  # Break out of the while loop

        if not query.strip():
            print("[ERROR] Query cannot be empty. Please enter a valid search query.")
            continue

        # Improve the query using LLM
        improved_query = improve_query_with_llm(query)
        if not improved_query.strip():
            print("[ERROR] The improved query is empty. Please try again with a different input.")
            continue
        print(f"[INFO] Improved search query: {improved_query}")

        # Perform searches with improved query
        print("[DEBUG] Starting search for query:", improved_query)
        google_results = google_search(improved_query)
        print(f"[DEBUG] Google search returned {len(google_results)} results.")
        bing_results = bing_search(improved_query)
        print(f"[DEBUG] Bing search returned {len(bing_results)} results.")

        # Extract query terms for relevance scoring
        query_terms = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        google_summaries = []
        bing_summaries = []

        print("\nSources:")
        # Process Google Search Results in parallel
        if google_results:
            print("[INFO] Processing Google results...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                google_futures = {executor.submit(fetch_print_summarize, result, query_terms): result for result in google_results}
                for future in as_completed(google_futures):
                    res = future.result()
                    google_summaries.append(res)
            print("[INFO] Finished processing Google results.")
        else:
            print("[WARN] No Google results to process.")

        # Process Bing Search Results in parallel
        if bing_results:
            print("[INFO] Processing Bing results...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                bing_futures = {executor.submit(fetch_print_summarize, result, query_terms): result for result in bing_results}
                for future in as_completed(bing_futures):
                    res = future.result()
                    bing_summaries.append(res)
            print("[INFO] Finished processing Bing results.")
        else:
            print("[WARN] No Bing results to process.")

        # Concatenate summaries from both search engines, if necessary
        all_summaries = ''.join(google_summaries + bing_summaries)
        print("[DEBUG] Total summaries length:", len(all_summaries))
        prompt = f"""You are an expert information analyst. Based on the following website summaries related to "{query}", provide a coherent, comprehensive answer.

If the query is a question, directly answer it first, then provide supporting context. If it's a topic search, provide a well-structured summary of the key information.

Requirements:
- Be factual and concise
- Organize information logically 
- Avoid repetition between sources
- Use only the information provided below
- Do not speculate or add external knowledge
- Write in a natural, flowing style rather than bullet points

Website summaries:
{all_summaries}
"""
        print("\n[INFO] Creating final summary...\n")
        sendToGPT(prompt)
        print("\n[INFO] Done.\n")
