import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.classes_ = None
        self.training_history = None
        self.top_features_per_class = None
    
    def preprocess_dataset(self, df):
        """Preprocess the training dataset"""
        try:
            # Check if required columns exist
            if 'Category' not in df.columns or 'Resume' not in df.columns:
                if 'category' in df.columns and 'text' in df.columns:
                    df = df.rename(columns={'category': 'Category', 'text': 'Resume'})
                elif 'category' in df.columns and 'resume' in df.columns:
                    df = df.rename(columns={'category': 'Category', 'resume': 'Resume'})
                else:
                    available_cols = list(df.columns)
                    raise ValueError(f"Required columns 'Category' and 'Resume' not found. Available columns: {available_cols}")
            
            # Remove null values
            df = df.dropna(subset=['Category', 'Resume'])
            
            # Clean text data
            df['clean_text'] = df['Resume'].apply(self._clean_text)
            
            # Remove empty texts
            df = df[df['clean_text'].str.len() > 0]
            
            return df
        except Exception as e:
            st.error(f"Error preprocessing dataset: {str(e)}")
            return None
    
    def _clean_text(self, text):
        """Clean text data"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the classification model"""
        try:
            # Preprocess dataset
            df_clean = self.preprocess_dataset(df)
            if df_clean is None:
                return False
            
            st.info(f"Training on {len(df_clean)} samples across {df_clean['Category'].nunique()} categories")
            
            # Prepare features and labels
            X = df_clean['clean_text']
            y = df_clean['Category']
            
            # Check if we have enough samples per class
            class_counts = y.value_counts()
            min_samples = class_counts.min()
            if min_samples < 2:
                st.warning(f"Some categories have very few samples (min: {min_samples}). Consider adding more data for better performance.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if min_samples >= 2 else None
            )
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            # Fit vectorizer and transform data
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Train logistic regression model
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state
            )
            
            self.model.fit(X_train_tfidf, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_tfidf)
            y_pred_proba = self.model.predict_proba(X_test_tfidf)
            
            # Store classes
            self.classes_ = self.model.classes_
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store training history
            self.training_history = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred, labels=self.classes_),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Extract top features per class
            self._extract_top_features()
            
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def _extract_top_features(self, n_features=20):
        """Extract top features for each class"""
        try:
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            self.top_features_per_class = {}
            
            for i, class_name in enumerate(self.classes_):
                # Get coefficients for this class
                coef = self.model.coef_[i]
                
                # Get top positive features (most indicative of this class)
                top_indices = np.argsort(coef)[-n_features:][::-1]
                top_features = feature_names[top_indices].tolist()
                
                self.top_features_per_class[class_name] = top_features
                
        except Exception as e:
            st.error(f"Error extracting top features: {str(e)}")
            self.top_features_per_class = {}
    
    def predict_resume_category(self, resume_text):
        """Predict category for a single resume"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        try:
            # Clean the text
            clean_text = self._clean_text(resume_text)
            
            # Vectorize
            text_tfidf = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities)
            
            # Get class probabilities dictionary
            class_probabilities = {
                class_name: prob 
                for class_name, prob in zip(self.classes_, probabilities)
            }
            
            return {
                'predicted_category': prediction,
                'confidence': confidence,
                'class_probabilities': class_probabilities
            }
            
        except Exception as e:
            st.error(f"Error predicting resume category: {str(e)}")
            return None
    
    def calculate_resume_score(self, resume_text, predicted_category):
        """Calculate comprehensive resume score"""
        try:
            clean_text = self._clean_text(resume_text)
            
            # Basic metrics
            word_count = len(clean_text.split())
            
            # Content richness score (normalized word count)
            richness_score = min(word_count / 400, 1.0)  # Assume 400 words is ideal
            
            # Keyword matching score
            if self.top_features_per_class and predicted_category in self.top_features_per_class:
                top_keywords = self.top_features_per_class[predicted_category]
                matched_keywords = [kw for kw in top_keywords if kw in clean_text.lower()]
                keyword_match_score = len(matched_keywords) / len(top_keywords)
            else:
                keyword_match_score = 0.0
            
            # Prediction result
            pred_result = self.predict_resume_category(resume_text)
            confidence_score = pred_result['confidence'] if pred_result else 0.0
            
            # Calculate final score (weighted combination)
            final_score = (
                0.5 * confidence_score +
                0.3 * richness_score +
                0.2 * keyword_match_score
            )
            
            return {
                'final_score': final_score,
                'confidence_score': confidence_score,
                'richness_score': richness_score,
                'keyword_match_score': keyword_match_score,
                'word_count': word_count,
                'matched_keywords': matched_keywords if 'matched_keywords' in locals() else []
            }
            
        except Exception as e:
            st.error(f"Error calculating resume score: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.model:
            return None
        
        return {
            'classes': list(self.classes_),
            'n_features': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            'training_history': self.training_history,
            'top_features_per_class': self.top_features_per_class
        }
