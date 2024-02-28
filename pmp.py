import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import fitz 
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Function to perform a Google search
def google_search(query, num_results=10):
    try:
        api_key = os.getenv('API_KEY')
        cse_key = os.getenv('CSE_KEY')
        # Build the resource object for Custom Search
        resource = build("customsearch", 'v1', developerKey=api_key).cse()

        # Execute the search query
        result = resource.list(q=query, cx=cse_key, num=num_results).execute()

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
def bing_search(query, num=10):
    headers = get_headers()
    bing_url = f"https://www.bing.com/search?q={query}"
    response = safe_request(bing_url, headers)
    if not response:
        return []
    return parse_bing_response(response, num)

# Helper Functions
def get_headers():
    return {'User-Agent': 'Mozilla/5.0'}

def safe_request(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError if status_code is not 200
        return response
    except requests.RequestException as e:
        print(f"Request to {url} failed: {e}")
        return None

def parse_bing_response(response, num):
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for result in soup.find_all('li', {'class': 'b_algo'}, limit=num):
        title_element = result.find('h2')
        link_element = result.find('a')
        if title_element and link_element:
            results.append({'title': title_element.text, 'link': link_element['href']})
    return results

# Add other functions for summarization, PDF processing, etc.
def read_pdf_content(response):
    with fitz.open(stream=response.raw, filetype="pdf") as doc:
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
    response = requests.get(url, headers=headers, stream=True)
    
    if url.endswith('.pdf') or is_pdf_response(response):
        # The content is a PDF file
        text = read_pdf_content(response)
        title = "PDF Document"
        main_image = "No main image"
    else:
        # The content is HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title = soup.title.string if soup.title else "No title"
        
        # Extract the text
        text = ' '.join([p.text for p in soup.find_all('p')])
        
        # Extract the main image
        main_image = soup.find('img')['src'] if soup.find('img') else None

    return {'title': title, 'text': text, 'main_image': main_image}

def summarize_text(text, n):
    sentences = nltk.sent_tokenize(text)
    word_counts = nltk.FreqDist(word for sentence in sentences for word in sentence.split())
    sentence_scores = []
    for sentence in sentences:
        score = 0
        for word in sentence.split():
            score += word_counts[word]
        sentence_scores.append(score / len(sentence.split()))  # Append the average score per word
    sorted_sentences = sorted(sentences, key=lambda sentence: sentence_scores[sentences.index(sentence)], reverse=True)
    summary = " ".join(sorted_sentences[:n])
    return summary

def fetch_and_summarize(result):
    try:
        content = read_link_content(result['link'])
        summarized_content = summarize_text(content["text"], 5)
        return summarized_content
    except Exception as exc:
        print(f"An error occurred for {result['link']}: {exc}")
        return ""

# Function to print the title and link, and return a summarized version
def fetch_print_summarize(item):
    try:
        title = item.get('title')
        link = item.get('link')
        print(link)
        # Fetch and summarize the content of the link
        summary = fetch_and_summarize({'link': link})
        return f"{title}, {link}\n{summary}\n"
    except Exception as exc:
        print(f'URL resulted in an exception: {exc}')
        return ""

def sendToGPT(copyPrompt):
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        #model="gpt-4",
        #model="gpt-3.5-turbo-1106",
        #response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": copyPrompt}
        ],
        stream=True
    )
    
    # Stream the response
    for chunk in response:
        if chunk.choices[0].delta.content is not None:  # Check for NoneType instead of "None"
            print(chunk.choices[0].delta.content, end='')  # Specify end parameter

# Example usage
if __name__ == "__main__":
    load_dotenv()
    nltk.download('punkt')

    while True:
        query = input("Please enter the search query: ")
        if query.lower() == "exit":
            print("Exiting the program.")
            break  # Break out of the while loop

        # Perform searches
        google_results = google_search(query)
        bing_results = bing_search(query)

        google_summaries = []
        bing_summaries = []

        print("\nSources:")
        # Process Google Search Results in parallel
        if google_results:
            with ThreadPoolExecutor(max_workers=10) as executor:
                google_futures = {executor.submit(fetch_print_summarize, result): result for result in google_results}
                for future in as_completed(google_futures):
                    google_summaries.append(future.result())

        # Process Bing Search Results in parallel
        if bing_results:
            with ThreadPoolExecutor(max_workers=10) as executor:
                bing_futures = {executor.submit(fetch_print_summarize, result): result for result in bing_results}
                
                for future in as_completed(bing_futures):
                    bing_summaries.append(future.result())

        # Concatenate summaries from both search engines, if necessary
        all_summaries = ''.join(google_summaries + bing_summaries)
        prompt = f"""Given the following summarized websites from around the world related to {query}, provide a concise overview that captures the key points from all websites. If the query is a question, respond directly and give support with the summary:

        {all_summaries}

        Please generate a comprehensive summary.

        Then, at the end provide a list of steps to accomplish {query} based on the summary.  
        """
        print("\n") # New line
        sendToGPT(prompt)
        print("\n") # New line
