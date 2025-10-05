import streamlit as st
import pandas as pd
import numpy as np
from utils.resume_processor import ResumeProcessor
from utils.visualizations import VisualizationHelper
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Resume Analysis - AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Analysis")

# Initialize processors
@st.cache_resource
def get_processors():
    return ResumeProcessor(), VisualizationHelper()

resume_processor, viz = get_processors()

# Check if model is trained
if 'trained_model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.error("❌ No trained model found. Please train a model first.")
    if st.button("🎯 Go to Model Training"):
        st.switch_page("pages/1_Model_Training.py")
    st.stop()

# Get model trainer from session state
if 'model_trainer' not in st.session_state:
    st.error("❌ Model trainer not found. Please retrain the model.")
    st.stop()

trainer = st.session_state.model_trainer

# Sidebar information
st.sidebar.markdown("### 🎯 Model Information")
model_info = trainer.get_model_info()
if model_info:
    st.sidebar.write(f"**Categories:** {len(model_info['classes'])}")
    st.sidebar.write(f"**Features:** {model_info['n_features']}")
    if model_info['training_history']:
        st.sidebar.write(f"**Accuracy:** {model_info['training_history']['accuracy']:.3f}")

st.sidebar.markdown("### 📋 Supported Formats")
st.sidebar.markdown("• PDF documents\n• Word documents (.docx)\n• Plain text files (.txt)")

# Main content
tab1, tab2 = st.tabs(["📤 Upload & Analyze", "📊 Analysis Results"])

with tab1:
    st.markdown("## 📤 Upload Resume for Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx', 'txt'],
        help="Upload a resume in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 📋 File Details")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            st.write(f"**File type:** {uploaded_file.type}")
        
        with col1:
            # Process resume
            with st.spinner("🔍 Processing resume..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Process the resume
                    resume_data = resume_processor.process_resume(uploaded_file, uploaded_file.name)
                    
                    if resume_data:
                        st.success("✅ Resume processed successfully!")
                        
                        # Store in session state for the results tab
                        st.session_state.current_resume = resume_data
                        
                        # Get prediction and scoring
                        prediction_result = trainer.predict_resume_category(resume_data['raw_text'])
                        score_result = trainer.calculate_resume_score(
                            resume_data['raw_text'], 
                            prediction_result['predicted_category']
                        )
                        
                        # Store results
                        st.session_state.prediction_result = prediction_result
                        st.session_state.score_result = score_result
                        
                        # Display quick results
                        st.markdown("### 🎯 Quick Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Predicted Role",
                                prediction_result['predicted_category']
                            )
                        
                        with col2:
                            st.metric(
                                "Confidence",
                                f"{prediction_result['confidence']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "Final Score",
                                f"{score_result['final_score']:.1%}"
                            )
                        
                        with col4:
                            st.metric(
                                "Word Count",
                                f"{resume_data['metrics']['word_count']:,}"
                            )
                        
                        # Progress bars for score breakdown
                        st.markdown("### 📊 Score Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Confidence Score**")
                            st.progress(prediction_result['confidence'])
                            
                            st.markdown("**Content Richness**")
                            st.progress(score_result['richness_score'])
                        
                        with col2:
                            st.markdown("**Keyword Match**")
                            st.progress(score_result['keyword_match_score'])
                            
                            st.markdown("**Final Score**")
                            st.progress(score_result['final_score'])
                        
                        # Switch to results tab button
                        if st.button("📊 View Detailed Analysis", type="primary"):
                            st.success("👆 Click on 'Analysis Results' tab to see detailed analysis!")
                    
                    else:
                        st.error("❌ Failed to process resume. Please check the file format and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing resume: {str(e)}")
    
    # Sample resume upload
    with st.expander("📝 Don't have a resume? Try a sample"):
        st.markdown("""
        You can create a simple text file with sample resume content like:
        
        ```
        John Doe
        Software Engineer
        
        Experience:
        - 5 years of Python development
        - Machine learning and data analysis
        - Web development with Django and Flask
        - Database design and optimization
        
        Skills:
        Python, JavaScript, SQL, Machine Learning, Data Science
        
        Education:
        Bachelor of Science in Computer Science
        ```
        """)

with tab2:
    st.markdown("## 📊 Detailed Analysis Results")
    
    if 'current_resume' not in st.session_state:
        st.info("👆 Upload and analyze a resume first to see detailed results here.")
    else:
        resume_data = st.session_state.current_resume
        prediction_result = st.session_state.prediction_result
        score_result = st.session_state.score_result
        
        # Header with filename
        st.markdown(f"### 📄 Analysis for: `{resume_data['filename']}`")
        
        # Main prediction results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎯 Prediction Results")
            
            # Category probabilities
            prob_df = pd.DataFrame([
                {'Category': cat, 'Probability': prob}
                for cat, prob in prediction_result['class_probabilities'].items()
            ]).sort_values('Probability', ascending=False)
            
            # Bar chart of probabilities
            fig = px.bar(
                prob_df.head(10),  # Top 10 categories
                x='Probability',
                y='Category',
                orientation='h',
                title='Category Prediction Probabilities',
                color='Probability',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Top Predictions")
            for i, row in prob_df.head(5).iterrows():
                confidence_color = "green" if row['Probability'] > 0.5 else "orange" if row['Probability'] > 0.2 else "red"
                st.markdown(
                    f"**{row['Category']}**: "
                    f"<span style='color: {confidence_color}'>{row['Probability']:.1%}</span>",
                    unsafe_allow_html=True
                )
        
        # Score visualization
        st.markdown("#### 🏆 Resume Score Analysis")
        score_fig = viz.create_resume_score_breakdown(score_result)
        st.plotly_chart(score_fig, use_container_width=True)
        
        # Content analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Content Metrics")
            metrics = resume_data['metrics']
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.metric("Word Count", f"{metrics['word_count']:,}")
                st.metric("Sentences", metrics['sentence_count'])
            
            with metric_col2:
                st.metric("Paragraphs", metrics['paragraph_count'])
                st.metric("Avg Word Length", f"{metrics['avg_word_length']:.1f}")
            
            # Readability score
            st.metric("Readability Score", f"{metrics['readability_score']:.1f}/100")
        
        with col2:
            st.markdown("#### 🔑 Top Keywords Found")
            keywords = resume_data['keywords'][:10]
            
            if keywords:
                # Create keyword frequency visualization
                keyword_df = pd.DataFrame({
                    'Keyword': keywords,
                    'Rank': range(1, len(keywords) + 1)
                })
                
                fig = px.bar(
                    keyword_df,
                    x='Rank',
                    y='Keyword',
                    orientation='h',
                    title='Top Keywords in Resume'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant keywords found.")
        
        # Keyword matching analysis
        if score_result['matched_keywords']:
            st.markdown("#### ✅ Matched Keywords")
            matched_kw = score_result['matched_keywords']
            
            # Display matched keywords as badges
            keyword_html = ""
            for kw in matched_kw[:15]:  # Show top 15
                keyword_html += f'<span style="background-color: #e6f3ff; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em;">{kw}</span> '
            
            st.markdown(keyword_html, unsafe_allow_html=True)
            
            st.success(f"✅ Found {len(matched_kw)} relevant keywords for {prediction_result['predicted_category']}")
        
        # Resume text preview
        with st.expander("📖 View Resume Content"):
            st.markdown("#### Original Text (First 1000 characters)")
            st.text_area(
                "Resume Content",
                value=resume_data['raw_text'][:1000] + ("..." if len(resume_data['raw_text']) > 1000 else ""),
                height=200,
                disabled=True
            )
            
            st.markdown("#### Processed Text (First 500 characters)")
            st.text_area(
                "Cleaned Content",
                value=resume_data['clean_text'][:500] + ("..." if len(resume_data['clean_text']) > 500 else ""),
                height=150,
                disabled=True
            )
        
        # Export results
        st.markdown("---")
        st.markdown("#### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create results summary
            results_summary = {
                'Filename': resume_data['filename'],
                'Predicted Category': prediction_result['predicted_category'],
                'Confidence Score': f"{prediction_result['confidence']:.3f}",
                'Final Score': f"{score_result['final_score']:.3f}",
                'Word Count': resume_data['metrics']['word_count'],
                'Keyword Matches': len(score_result['matched_keywords']),
                'Top Keywords': ', '.join(resume_data['keywords'][:5])
            }
            
            # Convert to DataFrame for CSV
            export_df = pd.DataFrame([results_summary])
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"resume_analysis_{resume_data['filename'].split('.')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create detailed JSON report
            detailed_report = {
                'resume_info': {
                    'filename': resume_data['filename'],
                    'metrics': resume_data['metrics']
                },
                'prediction': prediction_result,
                'scoring': score_result,
                'keywords': resume_data['keywords']
            }
            
            import json
            json_report = json.dumps(detailed_report, indent=2)
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json_report,
                file_name=f"resume_analysis_{resume_data['filename'].split('.')[0]}.json",
                mime="application/json",
                use_container_width=True
            )
