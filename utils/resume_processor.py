import os
import re
import numpy as np
import pandas as pd
import docx2txt
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

class ResumeProcessor:
    def __init__(self):
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def extract_text_from_file(self, file_obj, file_type):
        """Extract text from uploaded file"""
        try:
            if file_type == "pdf":
                return self._extract_from_pdf(file_obj)
            elif file_type == "docx":
                return self._extract_from_docx(file_obj)
            elif file_type == "txt":
                return self._extract_from_txt(file_obj)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            st.error(f"Error extracting text from file: {str(e)}")
            return ""
    
    def _extract_from_pdf(self, file_obj):
        """Extract text from PDF file"""
        text = ""
        try:
            reader = PyPDF2.PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
        return text
    
    def _extract_from_docx(self, file_obj):
        """Extract text from DOCX file"""
        try:
            # Save uploaded file temporarily to read with docx2txt
            temp_file = "temp_resume.docx"
            with open(temp_file, "wb") as f:
                f.write(file_obj.read())
            
            text = docx2txt.process(temp_file)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def _extract_from_txt(self, file_obj):
        """Extract text from TXT file"""
        try:
            # Handle both string and bytes
            content = file_obj.read()
            if isinstance(content, bytes):
                text = content.decode('utf-8', errors='ignore')
            else:
                text = content
            return text
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize, remove stopwords, and lemmatize
        words = []
        for word in text.split():
            if word not in self.stop_words and len(word) > 2:
                lemmatized_word = self.lemmatizer.lemmatize(word)
                words.append(lemmatized_word)
        
        return " ".join(words)
    
    def analyze_resume_content(self, text):
        """Analyze resume content and extract metrics"""
        if not text:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_word_length": 0,
                "readability_score": 0
            }
        
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraph_count = len([p for p in paragraphs if p.strip()])
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Simple readability score based on avg sentence length and word length
        avg_sentence_length = word_count / max(sentence_count, 1)
        readability_score = max(0, min(100, 100 - (avg_sentence_length * 2) - (avg_word_length * 5)))
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": round(avg_word_length, 2),
            "readability_score": round(readability_score, 2)
        }
    
    def get_top_keywords(self, text, top_n=10):
        """Extract top keywords from text"""
        if not text:
            return []
        
        words = text.split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def process_resume(self, file_obj, filename):
        """Main method to process a resume file"""
        file_extension = filename.split('.')[-1].lower()
        
        # Extract text
        raw_text = self.extract_text_from_file(file_obj, file_extension)
        
        if not raw_text:
            return None
        
        # Preprocess text
        clean_text = self.preprocess_text(raw_text)
        
        # Analyze content
        content_metrics = self.analyze_resume_content(raw_text)
        
        # Get keywords
        keywords = self.get_top_keywords(clean_text)
        
        return {
            "filename": filename,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "metrics": content_metrics,
            "keywords": keywords
        }
