import streamlit as st
import pandas as pd
import numpy as np
from utils.resume_processor import ResumeProcessor
from utils.visualizations import VisualizationHelper
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import time

# Page config
st.set_page_config(
    page_title="Batch Processing - AI Resume Screening",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Batch Resume Processing")

# Initialize processors
@st.cache_resource
def get_processors():
    return ResumeProcessor(), VisualizationHelper()

resume_processor, viz = get_processors()

# Check if model is trained
if 'trained_model' not in st.session_state or 'model_trainer' not in st.session_state:
    st.error("❌ No trained model found. Please train a model first.")
    if st.button("🎯 Go to Model Training"):
        st.switch_page("pages/1_Model_Training.py")
    st.stop()

trainer = st.session_state.model_trainer

# Sidebar
st.sidebar.markdown("### 📊 Batch Processing")
st.sidebar.markdown("""
**Features:**
- Multiple file upload
- Parallel processing
- Sortable results
- CSV export
- Visual analytics
""")

if 'batch_results' in st.session_state:
    st.sidebar.markdown(f"### 📈 Current Batch")
    st.sidebar.write(f"**Files Processed:** {len(st.session_state.batch_results)}")
    avg_score = np.mean([r['Final Score'] for r in st.session_state.batch_results])
    st.sidebar.write(f"**Average Score:** {avg_score:.2f}")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "📊 Results & Analysis", "📈 Visualizations"])

with tab1:
    st.markdown("## 📤 Batch Resume Upload")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple resumes for batch processing"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Display uploaded files
        with st.expander("📋 View Uploaded Files"):
            files_data = []
            for file in uploaded_files:
                files_data.append({
                    'Filename': file.name,
                    'Size (KB)': round(file.size / 1024, 2),
                    'Type': file.type
                })
            
            files_df = pd.DataFrame(files_data)
            st.dataframe(files_df, use_container_width=True)
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sort_by = st.selectbox(
                "Sort results by:",
                ["Final Score", "Confidence Score", "Word Count", "Filename"],
                help="Choose how to sort the results"
            )
        
        with col2:
            sort_order = st.selectbox(
                "Sort order:",
                ["Descending", "Ascending"]
            )
        
        # Process files button
        if st.button("🚀 Process All Resumes", type="primary", use_container_width=True):
            
            # Initialize results storage
            results = []
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Process resume
                    resume_data = resume_processor.process_resume(uploaded_file, uploaded_file.name)
                    
                    if resume_data:
                        # Get prediction
                        prediction_result = trainer.predict_resume_category(resume_data['raw_text'])
                        
                        # Get score
                        score_result = trainer.calculate_resume_score(
                            resume_data['raw_text'],
                            prediction_result['predicted_category']
                        )
                        
                        # Compile results
                        result = {
                            'Filename': resume_data['filename'],
                            'Predicted Category': prediction_result['predicted_category'],
                            'Confidence Score': prediction_result['confidence'],
                            'Final Score': score_result['final_score'],
                            'Word Count': resume_data['metrics']['word_count'],
                            'Keyword Matches': len(score_result['matched_keywords']),
                            'Content Richness': score_result['richness_score'],
                            'Keyword Match Score': score_result['keyword_match_score'],
                            'Top Keywords': ', '.join(resume_data['keywords'][:5])
                        }
                        
                        results.append(result)
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.1)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Complete processing
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            
            if results:
                # Store results in session state
                st.session_state.batch_results = results
                
                # Create DataFrame
                results_df = pd.DataFrame(results)
                
                # Sort results
                ascending = sort_order == "Ascending"
                results_df = results_df.sort_values(by=sort_by, ascending=ascending)
                
                st.success(f"🎉 Successfully processed {len(results)} out of {len(uploaded_files)} files!")
                
                # Show quick stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Files Processed", len(results))
                
                with col2:
                    avg_score = results_df['Final Score'].mean()
                    st.metric("Average Score", f"{avg_score:.2f}")
                
                with col3:
                    top_category = results_df['Predicted Category'].mode().iloc[0]
                    st.metric("Top Category", top_category)
                
                with col4:
                    high_confidence = (results_df['Confidence Score'] > 0.8).sum()
                    st.metric("High Confidence", f"{high_confidence}/{len(results)}")
                
            else:
                st.error("❌ No files were processed successfully.")

with tab2:
    st.markdown("## 📊 Results & Analysis")
    
    if 'batch_results' not in st.session_state or not st.session_state.batch_results:
        st.info("👆 Upload and process files first to see results here.")
    else:
        results = st.session_state.batch_results
        results_df = pd.DataFrame(results)
        
        # Sorting controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📈 Batch Results ({len(results)} resumes)")
        
        with col2:
            # Re-sort option
            sort_column = st.selectbox(
                "Sort by:",
                ["Final Score", "Confidence Score", "Word Count", "Keyword Matches"],
                key="results_sort"
            )
        
        # Sort dataframe
        display_df = results_df.sort_values(by=sort_column, ascending=False)
        
        # Format numeric columns for display
        display_df_formatted = display_df.copy()
        display_df_formatted['Confidence Score'] = display_df_formatted['Confidence Score'].apply(lambda x: f"{x:.1%}")
        display_df_formatted['Final Score'] = display_df_formatted['Final Score'].apply(lambda x: f"{x:.1%}")
        display_df_formatted['Content Richness'] = display_df_formatted['Content Richness'].apply(lambda x: f"{x:.1%}")
        display_df_formatted['Keyword Match Score'] = display_df_formatted['Keyword Match Score'].apply(lambda x: f"{x:.1%}")
        
        # Display results table
        st.dataframe(
            display_df_formatted,
            use_container_width=True,
            column_config={
                "Filename": st.column_config.TextColumn("📄 Filename", width="medium"),
                "Predicted Category": st.column_config.TextColumn("🎯 Category", width="medium"),
                "Final Score": st.column_config.TextColumn("🏆 Score", width="small"),
                "Confidence Score": st.column_config.TextColumn("📊 Confidence", width="small"),
                "Word Count": st.column_config.NumberColumn("📝 Words", width="small"),
                "Keyword Matches": st.column_config.NumberColumn("🔑 Keywords", width="small")
            }
        )
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Score Distribution")
            score_stats = display_df['Final Score'].describe()
            st.write(f"**Mean:** {score_stats['mean']:.2%}")
            st.write(f"**Median:** {score_stats['50%']:.2%}")
            st.write(f"**Std Dev:** {score_stats['std']:.2%}")
            st.write(f"**Min:** {score_stats['min']:.2%}")
            st.write(f"**Max:** {score_stats['max']:.2%}")
        
        with col2:
            st.markdown("#### Category Distribution")
            category_counts = display_df['Predicted Category'].value_counts()
            for category, count in category_counts.head(5).items():
                percentage = count / len(display_df) * 100
                st.write(f"**{category}:** {count} ({percentage:.1f}%)")
        
        with col3:
            st.markdown("#### Performance Tiers")
            high_score = (display_df['Final Score'] >= 0.8).sum()
            medium_score = ((display_df['Final Score'] >= 0.6) & (display_df['Final Score'] < 0.8)).sum()
            low_score = (display_df['Final Score'] < 0.6).sum()
            
            st.write(f"**High (≥80%):** {high_score} resumes")
            st.write(f"**Medium (60-79%):** {medium_score} resumes")
            st.write(f"**Low (<60%):** {low_score} resumes")
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full results CSV
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name="batch_resume_analysis_full.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Top performers CSV
            top_performers = display_df.head(10)
            top_csv = top_performers.to_csv(index=False)
            st.download_button(
                label="📥 Download Top 10 (CSV)",
                data=top_csv,
                file_name="batch_resume_analysis_top10.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Summary report
            summary_data = {
                'Metric': ['Total Resumes', 'Average Score', 'Top Category', 'High Score Count'],
                'Value': [
                    len(display_df),
                    f"{display_df['Final Score'].mean():.2%}",
                    display_df['Predicted Category'].mode().iloc[0],
                    (display_df['Final Score'] >= 0.8).sum()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=summary_csv,
                file_name="batch_resume_analysis_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab3:
    st.markdown("## 📈 Visual Analytics")
    
    if 'batch_results' not in st.session_state or not st.session_state.batch_results:
        st.info("👆 Process some resumes first to see visualizations here.")
    else:
        results_df = pd.DataFrame(st.session_state.batch_results)
        
        # Score distribution
        st.markdown("### 📊 Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of final scores
            fig_hist = px.histogram(
                results_df,
                x='Final Score',
                nbins=20,
                title='Distribution of Final Scores',
                labels={'Final Score': 'Final Score', 'count': 'Number of Resumes'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot of scores by category
            fig_box = px.box(
                results_df,
                x='Predicted Category',
                y='Final Score',
                title='Score Distribution by Category'
            )
            fig_box.update_xaxes(tickangle=45)
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Category analysis
        st.markdown("### 🎯 Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category pie chart
            category_fig = viz.create_category_pie_chart(results_df)
            if category_fig:
                st.plotly_chart(category_fig, use_container_width=True)
        
        with col2:
            # Category bar chart
            category_counts = results_df['Predicted Category'].value_counts()
            fig_bar = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title='Resume Count by Predicted Category',
                labels={'x': 'Number of Resumes', 'y': 'Category'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Correlation Analysis")
        
        # Scatter plot matrix
        numeric_cols = ['Confidence Score', 'Final Score', 'Word Count', 'Keyword Matches']
        correlation_data = results_df[numeric_cols]
        
        fig_scatter = px.scatter_matrix(
            correlation_data,
            title="Correlation Matrix of Resume Metrics",
            height=600
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance insights
        st.markdown("### 💡 Performance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score vs Word Count
            fig_scatter_wc = px.scatter(
                results_df,
                x='Word Count',
                y='Final Score',
                color='Predicted Category',
                title='Final Score vs Word Count',
                hover_data=['Filename']
            )
            fig_scatter_wc.update_layout(height=400)
            st.plotly_chart(fig_scatter_wc, use_container_width=True)
        
        with col2:
            # Confidence vs Final Score
            fig_scatter_conf = px.scatter(
                results_df,
                x='Confidence Score',
                y='Final Score',
                color='Predicted Category',
                title='Final Score vs Confidence Score',
                hover_data=['Filename']
            )
            fig_scatter_conf.update_layout(height=400)
            st.plotly_chart(fig_scatter_conf, use_container_width=True)
        
        # Top performers table
        st.markdown("### 🏆 Top Performers")
        
        top_performers = results_df.nlargest(5, 'Final Score')[
            ['Filename', 'Predicted Category', 'Final Score', 'Confidence Score', 'Word Count']
        ].copy()
        
        # Format scores as percentages
        top_performers['Final Score'] = top_performers['Final Score'].apply(lambda x: f"{x:.1%}")
        top_performers['Confidence Score'] = top_performers['Confidence Score'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            top_performers,
            use_container_width=True,
            column_config={
                "Filename": "📄 Resume",
                "Predicted Category": "🎯 Category",
                "Final Score": "🏆 Score",
                "Confidence Score": "📊 Confidence",
                "Word Count": "📝 Words"
            }
        )
        
        # Clear results button
        st.markdown("---")
        if st.button("🗑️ Clear Results", type="secondary"):
            if st.button("⚠️ Confirm Clear"):
                del st.session_state.batch_results
                st.success("✅ Results cleared!")
                st.rerun()
