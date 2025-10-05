import streamlit as st
import pandas as pd
import numpy as np
from utils.model_trainer import ModelTrainer
from utils.visualizations import VisualizationHelper
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Model Training - AI Resume Screening",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Model Training")

# Initialize classes
@st.cache_resource
def get_trainer():
    return ModelTrainer()

@st.cache_resource
def get_visualizer():
    return VisualizationHelper()

trainer = get_trainer()
viz = get_visualizer()

# Sidebar for navigation
st.sidebar.markdown("### Training Steps")
st.sidebar.markdown("""
1. **Upload Dataset** 📁
2. **Configure Training** ⚙️
3. **Train Model** 🚀
4. **Evaluate Performance** 📊
""")

# Main content
tab1, tab2, tab3 = st.tabs(["📁 Dataset Upload", "🚀 Train Model", "📊 Model Performance"])

with tab1:
    st.markdown("## 📁 Dataset Upload")
    
    st.markdown("""
    Upload a CSV file containing resume data with the following columns:
    - **Category**: Job category/role (e.g., 'Data Scientist', 'Software Engineer')
    - **Resume**: Resume text content
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with 'Category' and 'Resume' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load and display dataset
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show dataset info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Dataset Overview")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Column Names:** {list(df.columns)}")
            
            with col2:
                st.markdown("### Sample Data")
                st.dataframe(df.head(), use_container_width=True)
            
            # Show category distribution
            if 'Category' in df.columns:
                category_counts = df['Category'].value_counts()
                
                st.markdown("### Category Distribution")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Resume Count by Category"
                    )
                    fig.update_layout(
                        xaxis_title="Number of Resumes",
                        yaxis_title="Category",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Category Counts:**")
                    for category, count in category_counts.items():
                        st.write(f"• {category}: {count}")
                        
                    # Store dataset in session state
                    st.session_state.training_dataset = df
                    
            else:
                st.error("❌ 'Category' column not found in the dataset")
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    # Sample dataset format
    with st.expander("📋 View Sample Dataset Format"):
        sample_data = pd.DataFrame({
            'Category': ['Data Scientist', 'Software Engineer', 'Marketing Manager'],
            'Resume': [
                'Python machine learning data analysis sklearn pandas...',
                'Java programming software development spring boot...',
                'Digital marketing social media campaign management...'
            ]
        })
        st.dataframe(sample_data, use_container_width=True)

with tab2:
    st.markdown("## 🚀 Train Model")
    
    if 'training_dataset' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first in the 'Dataset Upload' tab.")
    else:
        df = st.session_state.training_dataset
        
        # Training configuration
        st.markdown("### ⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                help="Percentage of data to use for testing"
            ) / 100
        
        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=1,
                max_value=100,
                value=42,
                help="Seed for reproducible results"
            )
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                status_text.text("Preprocessing dataset...")
                progress_bar.progress(20)
                
                # Train model
                success = trainer.train_model(df, test_size=test_size, random_state=random_state)
                
                if success:
                    progress_bar.progress(100)
                    status_text.text("Training completed successfully!")
                    
                    # Store model in session state
                    st.session_state.trained_model = trainer.model
                    st.session_state.vectorizer = trainer.vectorizer
                    st.session_state.top_features = trainer.top_features_per_class
                    st.session_state.model_trainer = trainer
                    
                    st.success("🎉 Model training completed successfully!")
                    
                    # Show basic metrics
                    model_info = trainer.get_model_info()
                    training_history = model_info['training_history']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{training_history['accuracy']:.3f}")
                    
                    with col2:
                        st.metric("Classes", len(model_info['classes']))
                    
                    with col3:
                        st.metric("Features", model_info['n_features'])
                    
                    with col4:
                        st.metric("Test Size", training_history['test_size'])
                    
                else:
                    progress_bar.progress(0)
                    status_text.text("Training failed!")
                    st.error("❌ Model training failed. Please check your dataset and try again.")

with tab3:
    st.markdown("## 📊 Model Performance")
    
    if 'model_trainer' not in st.session_state:
        st.warning("⚠️ No trained model found. Please train a model first.")
    else:
        trainer = st.session_state.model_trainer
        model_info = trainer.get_model_info()
        
        if model_info and model_info['training_history']:
            training_history = model_info['training_history']
            
            # Performance metrics
            st.markdown("### 📈 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Accuracy",
                    f"{training_history['accuracy']:.3f}",
                    delta=f"{(training_history['accuracy'] - 0.5):.3f}"
                )
            
            with col2:
                avg_precision = np.mean([
                    metrics['precision'] for class_name, metrics in training_history['classification_report'].items()
                    if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']
                ])
                st.metric("Average Precision", f"{avg_precision:.3f}")
            
            with col3:
                avg_recall = np.mean([
                    metrics['recall'] for class_name, metrics in training_history['classification_report'].items()
                    if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']
                ])
                st.metric("Average Recall", f"{avg_recall:.3f}")
            
            # Visualizations
            st.markdown("### 📊 Detailed Analysis")
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            confusion_fig = viz.create_confusion_matrix(
                training_history['confusion_matrix'],
                list(trainer.classes_)
            )
            st.plotly_chart(confusion_fig, use_container_width=True)
            
            # Classification Report
            st.markdown("#### Performance by Category")
            classification_fig = viz.create_classification_report_chart(
                training_history['classification_report']
            )
            st.plotly_chart(classification_fig, use_container_width=True)
            
            # Class Distribution
            st.markdown("#### Prediction Distribution")
            distribution_fig = viz.create_class_distribution(
                training_history['y_test'],
                training_history['y_pred'],
                list(trainer.classes_)
            )
            st.plotly_chart(distribution_fig, use_container_width=True)
            
            # Confidence Distribution
            st.markdown("#### Model Confidence")
            confidence_fig = viz.create_confidence_distribution(
                training_history['y_pred_proba'],
                list(trainer.classes_)
            )
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Feature Importance
            st.markdown("#### Top Keywords by Category")
            if model_info['top_features_per_class']:
                selected_class = st.selectbox(
                    "Select Category to View Top Keywords:",
                    list(model_info['top_features_per_class'].keys())
                )
                
                if selected_class:
                    feature_fig = viz.create_feature_importance_chart(
                        model_info['top_features_per_class'],
                        selected_class
                    )
                    if feature_fig:
                        st.plotly_chart(feature_fig, use_container_width=True)
                    
                    # Show keywords as text
                    st.markdown(f"**Top keywords for {selected_class}:**")
                    keywords = model_info['top_features_per_class'][selected_class][:10]
                    st.write(" • ".join(keywords))

# Model management
st.markdown("---")
st.markdown("## 🔧 Model Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save Model", use_container_width=True):
        if 'trained_model' in st.session_state:
            st.success("✅ Model saved to session state!")
        else:
            st.error("❌ No model to save!")

with col2:
    if st.button("🗑️ Clear Model", use_container_width=True):
        if st.button("⚠️ Confirm Clear", use_container_width=True):
            for key in ['trained_model', 'vectorizer', 'top_features', 'model_trainer']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Model cleared!")
            st.rerun()
