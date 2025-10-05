# AI Resume Screening System

## Overview

This is an AI-powered resume screening and classification system built with Streamlit. The application uses machine learning (specifically Logistic Regression with TF-IDF vectorization) to automatically categorize resumes into job roles and rank candidates based on AI-derived confidence scores. The system supports single resume analysis, batch processing of multiple resumes, and provides comprehensive visualizations of model performance and insights.

Key capabilities:
- Train custom classification models on labeled resume datasets
- Extract and process text from PDF, DOCX, and TXT resume files
- Classify resumes into predefined job categories
- Score and rank candidates based on model confidence
- Batch process multiple resumes simultaneously
- Visualize model performance metrics and feature importance
- Export results to CSV for further analysis

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Decision:** Streamlit-based multi-page application
- **Rationale:** Streamlit provides rapid development for ML/data science applications with built-in state management and interactive components
- **Structure:** Main dashboard (app.py) with separate pages for distinct workflows:
  - Model Training (pages/1_Model_Training.py)
  - Single Resume Analysis (pages/2_Resume_Analysis.py)
  - Batch Processing (pages/3_Batch_Processing.py)
  - Advanced Visualizations (pages/4_Visualizations.py)
- **State Management:** Leverages Streamlit's session_state for persisting trained models, vectorizers, and processing results across page navigation
- **Pros:** Fast development, automatic UI reactivity, built-in caching
- **Cons:** Limited customization compared to traditional web frameworks, requires server-side execution

### Machine Learning Pipeline
**Decision:** Scikit-learn with TF-IDF + Logistic Regression
- **Rationale:** Proven text classification approach that balances performance with interpretability for resume screening
- **Components:**
  - TfidfVectorizer for feature extraction from resume text
  - LogisticRegression for multi-class classification
  - LabelEncoder for category encoding
- **Workflow:** 
  1. Dataset upload and preprocessing
  2. Text cleaning (lowercase, punctuation removal, stopword removal, lemmatization)
  3. TF-IDF vectorization
  4. Model training with train/test split
  5. Performance evaluation with classification reports and confusion matrices
- **Pros:** Fast training, interpretable features, good baseline performance
- **Cons:** May not capture semantic relationships as well as deep learning approaches

### Text Processing Architecture
**Decision:** NLTK-based preprocessing pipeline
- **Rationale:** NLTK provides mature, reliable NLP tools for text normalization
- **Pipeline Steps:**
  1. File format detection and text extraction (PDF via PyPDF2, DOCX via docx2txt, TXT direct read)
  2. Lowercasing
  3. Punctuation and number removal
  4. Stopword filtering using NLTK's English corpus
  5. Lemmatization using WordNetLemmatizer
- **Caching:** NLTK data (punkt, stopwords, wordnet) downloaded on initialization
- **Pros:** Robust preprocessing, reduces noise in text data
- **Cons:** Processing time increases with document size

### Visualization System
**Decision:** Plotly for interactive visualizations
- **Rationale:** Plotly provides highly interactive, publication-quality charts that integrate seamlessly with Streamlit
- **Visualization Types:**
  - Confusion matrices for model performance
  - Bar charts for classification metrics (precision, recall, F1-score)
  - Feature importance charts showing top TF-IDF features per category
  - Distribution plots for candidate scores
  - Category distribution pie/bar charts
- **Theming:** Custom CSS with gradient color schemes supporting light/dark modes
- **Pros:** Interactive tooltips, zoom capabilities, professional appearance
- **Cons:** Larger bundle size than static plotting libraries

### Data Flow Architecture
**Decision:** In-memory processing with session state persistence
- **Rationale:** Suitable for prototype/demo applications without database overhead
- **Flow:**
  1. Upload dataset → Validate columns → Preprocess text → Train model → Store in session_state
  2. Upload resume(s) → Extract text → Preprocess → Vectorize → Predict → Display results
  3. Batch processing streams through uploaded files sequentially
- **Limitations:** Data lost on session end, not suitable for production multi-user scenarios
- **Future Consideration:** Could add database persistence for trained models and processing history

### Modular Utility Design
**Decision:** Separate utility classes for distinct responsibilities
- **Components:**
  - `ModelTrainer`: Handles dataset preprocessing, model training, evaluation
  - `ResumeProcessor`: Text extraction from multiple formats, text cleaning
  - `VisualizationHelper`: Chart generation with consistent styling
  - `theme_helper`: CSS generation and theme management
- **Rationale:** Separation of concerns, reusability, easier testing
- **Caching:** Utility classes cached with `@st.cache_resource` to avoid reinitialization

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework and UI rendering
- **pandas**: Data manipulation and DataFrame operations for datasets
- **numpy**: Numerical operations and array handling
- **scikit-learn**: Machine learning algorithms (Logistic Regression, TF-IDF, metrics)
- **nltk**: Natural language processing (tokenization, stopwords, lemmatization)
- **plotly**: Interactive visualization library (plotly.express, plotly.graph_objects)
- **PyPDF2**: PDF text extraction
- **docx2txt**: Microsoft Word document text extraction
- **matplotlib** and **seaborn**: Referenced in attached code (may be legacy, not actively used in current codebase)

### NLTK Data Dependencies
The application automatically downloads required NLTK corpora:
- **punkt**: Sentence and word tokenization
- **stopwords**: English stopword list for filtering
- **wordnet**: Lexical database for lemmatization

### File Format Support
- **PDF**: Binary format, extracted via PyPDF2
- **DOCX**: Microsoft Word format, extracted via docx2txt
- **TXT**: Plain text format, direct read
- **CSV**: Dataset format for training data upload

### No External APIs or Databases
The application currently operates standalone without:
- External API calls (all processing local)
- Database connections (in-memory session storage only)
- Cloud storage integrations
- Authentication services

This architecture makes the system self-contained and easy to deploy but limits scalability and multi-user capabilities.