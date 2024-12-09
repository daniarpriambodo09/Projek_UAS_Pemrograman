import os
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
import nltk
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Download necessary NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')  # Adding punkt_tab download explicitly
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

# Define the output directory
OUTPUT_DIR = '/opt/airflow/dags/'

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting text from URL: {e}"

def preprocess_text(text):
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize into words
    words = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words and word not in punctuation]
    
    return sentences, words

def score_sentences(sentences, word_freq):
    sentence_scores = {}
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score = 0
        for word in words:
            if word in word_freq:
                score += word_freq[word]
        sentence_scores[sentence] = score
    
    return sentence_scores

def generate_summary(text, num_sentences=3):
    # Preprocess text
    sentences, words = preprocess_text(text)
    
    # Calculate word frequencies
    word_freq = FreqDist(words)
    
    # Score sentences
    sentence_scores = score_sentences(sentences, word_freq)
    
    # Get top sentences
    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Sort sentences by their original order
    summary_sentences.sort(key=lambda x: sentences.index(x[0]))
    
    # Join sentences
    summary = ' '.join([sentence[0] for sentence in summary_sentences])
    
    return summary

def save_to_txt(data, output_filename):
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)
    print(f"Data saved to {file_path}")

def combine_and_save_files(original_filename, summary_filename, output_filename):
    original_filepath = os.path.join(OUTPUT_DIR, original_filename)
    summary_filepath = os.path.join(OUTPUT_DIR, summary_filename)
    
    # Check if the original text file exists
    if not os.path.exists(original_filepath):
        print(f"Error: {original_filepath} does not exist!")
        return
    
    # Read original text
    with open(original_filepath, 'r', encoding='utf-8') as file:
        original_text = file.read().strip()
    
    # Read summary text
    summary_filepath = os.path.join(OUTPUT_DIR, summary_filename)
    with open(summary_filepath, 'r', encoding='utf-8') as file:
        summary_text = file.read().strip()
    
    # Combine both texts into one paragraph
    combined_text = ' '.join([original_text, summary_text])
    
    # Save combined text into one file
    save_to_txt(combined_text, output_filename)


def process_article_and_save():
    url = "https://www.kompas.com/tren/read/2024/12/02/154953665/kasus-donasi-agus-salim-dan-pengkhianatan-kepercayaan?page=all"
    original_text = extract_text_from_url(url)

    if "Error" not in original_text:
        summary_text = generate_summary(original_text, num_sentences=3)
        
        # Save original text to TXT
        save_to_txt(original_text, "original_text.txt")
        
        # Save summary text to TXT
        save_to_txt(summary_text, "summary_text.txt")
        
        # Combine both files into one file
        combine_and_save_files("original_text.txt", "summary_text.txt", "combined_text.txt")
    else:
        print(original_text)

# Define the DAG
with DAG(
    'article_summarization_dag',
    default_args={'owner': 'airflow'},
    description='Extract and summarize article',
    schedule_interval=None,  # Set to None for manual triggering or set a cron schedule
    start_date=datetime(2024, 12, 5),
    catchup=False,
) as dag:

    # Task to download NLTK data
    download_nltk_task = PythonOperator(
        task_id='download_nltk_data',
        python_callable=download_nltk_data,
    )

    # Task to process the article
    summarize_task = PythonOperator(
        task_id='process_article',
        python_callable=process_article_and_save,
    )

    # Set task dependencies
    download_nltk_task >> summarize_task
