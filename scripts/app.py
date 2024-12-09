import streamlit as st
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return "Error extracting text from URL"

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

def count_words(text):
    return len(word_tokenize(text))

# Streamlit UI
st.title('Web Text Summarizer')

url = st.text_input('Masukkan URL artikel (contoh: kompas.com, detik.com):', '')

if st.button('Summarize'):
    if url:
        # Extract text from URL
        original_text = extract_text_from_url(url)
        
        if original_text != "Error extracting text from URL":
            # Generate summary
            summary = generate_summary(original_text)
            
            # Count words
            original_word_count = count_words(original_text)
            summary_word_count = count_words(summary)
            
            # Display results
            st.header('Teks Original:')
            st.write(original_text)
            st.write(f'Jumlah kata: {original_word_count}')
            
            st.header('Hasil Summarization:')
            st.write(summary)
            st.write(f'Jumlah kata: {summary_word_count}')
            
            # Display reduction percentage
            reduction = ((original_word_count - summary_word_count) / original_word_count) * 100
            st.write(f'Pengurangan teks: {reduction:.2f}%')
        else:
            st.error('Gagal mengekstrak teks dari URL. Pastikan URL valid dan dapat diakses.')
    else:
        st.warning('Silakan masukkan URL terlebih dahulu.')