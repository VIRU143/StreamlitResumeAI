import streamlit as st
import pandas as pd
import numpy as np
from utils.visualizations import VisualizationHelper
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Visualizations - AI Resume Screening",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Advanced Analytics & Visualizations")

# Initialize visualization helper
@st.cache_resource
def get_visualizer():
    return VisualizationHelper()

viz = get_visualizer()

# Check if model is trained
if 'model_trainer' not in st.session_state:
    st.error("❌ No trained model found. Please train a model first.")
    if st.button("🎯 Go to Model Training"):
        st.switch_page("pages/1_Model_Training.py")
    st.stop()

trainer = st.session_state.model_trainer
model_info = trainer.get_model_info()

# Sidebar navigation
st.sidebar.markdown("### 📊 Visualization Types")
viz_options = [
    "📈 Model Performance",
    "🔍 Feature Analysis", 
    "📄 Resume Insights",
    "📊 Batch Analytics"
]

if 'batch_results' in st.session_state:
    st.sidebar.success(f"✅ Batch data available ({len(st.session_state.batch_results)} resumes)")
else:
    st.sidebar.info("ℹ️ Run batch processing to unlock more visualizations")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Performance", "🔍 Feature Analysis", "📄 Resume Analytics", "📊 Interactive Dashboard"])

with tab1:
    st.markdown("## 📈 Model Performance Analysis")
    
    if not model_info or not model_info.get('training_history'):
        st.warning("⚠️ No training history available. Please retrain the model to see performance metrics.")
    else:
        training_history = model_info['training_history']
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", f"{training_history['accuracy']:.3f}")
        
        with col2:
            # Calculate average precision
            avg_precision = np.mean([
                metrics['precision'] for class_name, metrics in training_history['classification_report'].items()
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']
            ])
            st.metric("Avg Precision", f"{avg_precision:.3f}")
        
        with col3:
            # Calculate average recall
            avg_recall = np.mean([
                metrics['recall'] for class_name, metrics in training_history['classification_report'].items()
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']
            ])
            st.metric("Avg Recall", f"{avg_recall:.3f}")
        
        with col4:
            st.metric("Classes", len(trainer.classes_))
        
        # Detailed visualizations
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Confusion Matrix")
            confusion_fig = viz.create_confusion_matrix(
                training_history['confusion_matrix'],
                list(trainer.classes_)
            )
            st.plotly_chart(confusion_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Performance by Category")
            classification_fig = viz.create_classification_report_chart(
                training_history['classification_report']
            )
            st.plotly_chart(classification_fig, use_container_width=True)
        
        # Class distribution comparison
        st.markdown("### 📈 Prediction vs Actual Distribution")
        distribution_fig = viz.create_class_distribution(
            training_history['y_test'],
            training_history['y_pred'],
            list(trainer.classes_)
        )
        st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Confidence analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎲 Confidence Distribution")
            confidence_fig = viz.create_confidence_distribution(
                training_history['y_pred_proba'],
                list(trainer.classes_)
            )
            st.plotly_chart(confidence_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Detailed Classification Report")
            
            # Create a formatted classification report table
            report_data = []
            for class_name, metrics in training_history['classification_report'].items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_data.append({
                        'Category': class_name,
                        'Precision': f"{metrics['precision']:.3f}",
                        'Recall': f"{metrics['recall']:.3f}",
                        'F1-Score': f"{metrics['f1-score']:.3f}",
                        'Support': int(metrics['support'])
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df, use_container_width=True)

with tab2:
    st.markdown("## 🔍 Feature Analysis")
    
    if not model_info.get('top_features_per_class'):
        st.warning("⚠️ No feature analysis available. Please retrain the model.")
    else:
        top_features = model_info['top_features_per_class']
        
        # Category selection
        selected_category = st.selectbox(
            "🎯 Select Category for Analysis:",
            list(top_features.keys()),
            help="Choose a category to analyze its top keywords"
        )
        
        if selected_category:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Feature importance chart
                st.markdown(f"### 📊 Top Keywords for {selected_category}")
                feature_fig = viz.create_feature_importance_chart(
                    top_features,
                    selected_category
                )
                if feature_fig:
                    st.plotly_chart(feature_fig, use_container_width=True)
            
            with col2:
                st.markdown(f"### 🔑 Keyword List")
                keywords = top_features[selected_category][:20]
                
                # Display as numbered list
                for i, keyword in enumerate(keywords, 1):
                    importance_color = "green" if i <= 5 else "orange" if i <= 10 else "gray"
                    st.markdown(
                        f"{i:2d}. <span style='color: {importance_color}; font-weight: bold;'>{keyword}</span>",
                        unsafe_allow_html=True
                    )
        
        # Cross-category keyword analysis
        st.markdown("---")
        st.markdown("### 🔄 Cross-Category Keyword Analysis")
        
        # Find common keywords across categories
        all_keywords = {}
        for category, keywords in top_features.items():
            for keyword in keywords[:10]:  # Top 10 from each category
                if keyword not in all_keywords:
                    all_keywords[keyword] = []
                all_keywords[keyword].append(category)
        
        # Find keywords that appear in multiple categories
        common_keywords = {kw: cats for kw, cats in all_keywords.items() if len(cats) > 1}
        
        if common_keywords:
            st.markdown("#### 🔗 Keywords Common Across Categories")
            
            common_data = []
            for keyword, categories in common_keywords.items():
                common_data.append({
                    'Keyword': keyword,
                    'Categories': ', '.join(categories),
                    'Count': len(categories)
                })
            
            common_df = pd.DataFrame(common_data).sort_values('Count', ascending=False)
            st.dataframe(common_df.head(15), use_container_width=True)
        
        # Category similarity analysis
        st.markdown("#### 📊 Category Keyword Overlap Heatmap")
        
        # Calculate keyword overlap between categories
        categories = list(top_features.keys())
        n_categories = len(categories)
        overlap_matrix = np.zeros((n_categories, n_categories))
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i != j:
                    set1 = set(top_features[cat1][:15])  # Top 15 keywords
                    set2 = set(top_features[cat2][:15])
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                    overlap_matrix[i][j] = overlap
                else:
                    overlap_matrix[i][j] = 1.0
        
        # Create heatmap
        fig_overlap = px.imshow(
            overlap_matrix,
            labels=dict(x="Category", y="Category", color="Overlap Score"),
            x=categories,
            y=categories,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Category Keyword Overlap Matrix"
        )
        
        # Add text annotations
        for i in range(n_categories):
            for j in range(n_categories):
                fig_overlap.add_annotation(
                    x=j, y=i,
                    text=f"{overlap_matrix[i][j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if overlap_matrix[i][j] > 0.5 else "black")
                )
        
        fig_overlap.update_layout(height=500)
        st.plotly_chart(fig_overlap, use_container_width=True)

with tab3:
    st.markdown("## 📄 Resume Analytics")
    
    # Individual resume analysis
    if 'current_resume' in st.session_state and 'score_result' in st.session_state:
        st.markdown("### 📋 Current Resume Analysis")
        
        resume_data = st.session_state.current_resume
        score_result = st.session_state.score_result
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Score breakdown visualization
            score_fig = viz.create_resume_score_breakdown(score_result)
            st.plotly_chart(score_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Resume Metrics")
            
            metrics = resume_data['metrics']
            st.write(f"**Word Count:** {metrics['word_count']:,}")
            st.write(f"**Sentences:** {metrics['sentence_count']}")
            st.write(f"**Paragraphs:** {metrics['paragraph_count']}")
            st.write(f"**Readability:** {metrics['readability_score']:.1f}/100")
            
            st.markdown("#### 🎯 Matching Analysis")
            st.write(f"**Keywords Found:** {len(score_result['matched_keywords'])}")
            st.write(f"**Match Rate:** {score_result['keyword_match_score']:.1%}")
            st.write(f"**Content Score:** {score_result['richness_score']:.1%}")
    
    # Batch analytics
    if 'batch_results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Batch Resume Analytics")
        
        results_df = pd.DataFrame(st.session_state.batch_results)
        
        # Score distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Score histogram with quartiles
            fig_hist = px.histogram(
                results_df,
                x='Final Score',
                nbins=25,
                title='Resume Score Distribution',
                marginal="box"
            )
            
            # Add quartile lines
            q1, q2, q3 = results_df['Final Score'].quantile([0.25, 0.5, 0.75])
            fig_hist.add_vline(x=q1, line_dash="dash", line_color="red", annotation_text="Q1")
            fig_hist.add_vline(x=q2, line_dash="dash", line_color="green", annotation_text="Median")
            fig_hist.add_vline(x=q3, line_dash="dash", line_color="red", annotation_text="Q3")
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Word count vs score analysis
            fig_scatter = px.scatter(
                results_df,
                x='Word Count',
                y='Final Score',
                color='Predicted Category',
                size='Confidence Score',
                hover_data=['Filename'],
                title='Score vs Word Count by Category'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance insights
        st.markdown("### 💡 Performance Insights")
        
        # Calculate insights
        high_performers = results_df[results_df['Final Score'] >= 0.8]
        avg_score_by_category = results_df.groupby('Predicted Category')['Final Score'].mean().sort_values(ascending=False)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏆 Top Performing Categories")
            for category, avg_score in avg_score_by_category.head(5).items():
                st.write(f"**{category}:** {avg_score:.1%}")
        
        with col2:
            st.markdown("#### 📊 Score Distribution")
            excellent = (results_df['Final Score'] >= 0.9).sum()
            good = ((results_df['Final Score'] >= 0.7) & (results_df['Final Score'] < 0.9)).sum()
            fair = ((results_df['Final Score'] >= 0.5) & (results_df['Final Score'] < 0.7)).sum()
            poor = (results_df['Final Score'] < 0.5).sum()
            
            st.write(f"**Excellent (≥90%):** {excellent}")
            st.write(f"**Good (70-89%):** {good}")
            st.write(f"**Fair (50-69%):** {fair}")
            st.write(f"**Needs Improvement (<50%):** {poor}")
        
        with col3:
            st.markdown("#### 🔍 Quality Indicators")
            avg_word_count = results_df['Word Count'].mean()
            avg_keywords = results_df['Keyword Matches'].mean()
            high_confidence = (results_df['Confidence Score'] >= 0.8).sum()
            
            st.write(f"**Avg Word Count:** {avg_word_count:,.0f}")
            st.write(f"**Avg Keywords:** {avg_keywords:.1f}")
            st.write(f"**High Confidence:** {high_confidence}/{len(results_df)}")

with tab4:
    st.markdown("## 📊 Interactive Dashboard")
    
    if 'batch_results' not in st.session_state:
        st.info("📁 Run batch processing first to access the interactive dashboard.")
    else:
        results_df = pd.DataFrame(st.session_state.batch_results)
        
        # Dashboard controls
        st.markdown("### 🎛️ Dashboard Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.slider(
                "Minimum Score Filter",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Filter resumes by minimum final score"
            )
        
        with col2:
            selected_categories = st.multiselect(
                "Category Filter",
                options=results_df['Predicted Category'].unique(),
                default=results_df['Predicted Category'].unique(),
                help="Select categories to display"
            )
        
        with col3:
            chart_type = st.selectbox(
                "Chart Type",
                ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Chart"],
                help="Choose visualization type"
            )
        
        # Filter data
        filtered_df = results_df[
            (results_df['Final Score'] >= min_score) &
            (results_df['Predicted Category'].isin(selected_categories))
        ]
        
        if filtered_df.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
        else:
            st.success(f"✅ Showing {len(filtered_df)} out of {len(results_df)} resumes")
            
            # Main visualization
            st.markdown("### 📈 Dynamic Visualization")
            
            if chart_type == "Scatter Plot":
                fig = px.scatter(
                    filtered_df,
                    x='Confidence Score',
                    y='Final Score',
                    color='Predicted Category',
                    size='Word Count',
                    hover_data=['Filename', 'Keyword Matches'],
                    title="Interactive Score Analysis"
                )
            
            elif chart_type == "Box Plot":
                fig = px.box(
                    filtered_df,
                    x='Predicted Category',
                    y='Final Score',
                    title="Score Distribution by Category"
                )
                fig.update_xaxes(tickangle=45)
            
            elif chart_type == "Violin Plot":
                fig = px.violin(
                    filtered_df,
                    x='Predicted Category',
                    y='Final Score',
                    title="Score Distribution (Violin Plot)"
                )
                fig.update_xaxes(tickangle=45)
            
            else:  # Bar Chart
                category_stats = filtered_df.groupby('Predicted Category').agg({
                    'Final Score': 'mean',
                    'Confidence Score': 'mean',
                    'Word Count': 'mean'
                }).round(3)
                
                fig = px.bar(
                    category_stats,
                    y=category_stats.index,
                    x='Final Score',
                    orientation='h',
                    title="Average Final Score by Category"
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics for filtered data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Filtered Count", len(filtered_df))
            
            with col2:
                st.metric("Average Score", f"{filtered_df['Final Score'].mean():.2%}")
            
            with col3:
                st.metric("Top Category", filtered_df['Predicted Category'].mode().iloc[0] if len(filtered_df) > 0 else "N/A")
            
            with col4:
                st.metric("High Performers", f"{(filtered_df['Final Score'] >= 0.8).sum()}/{len(filtered_df)}")
            
            # Detailed filtered results
            with st.expander("📋 View Filtered Results"):
                display_cols = ['Filename', 'Predicted Category', 'Final Score', 'Confidence Score', 'Word Count']
                
                # Format the display dataframe
                display_df = filtered_df[display_cols].copy()
                display_df['Final Score'] = display_df['Final Score'].apply(lambda x: f"{x:.1%}")
                display_df['Confidence Score'] = display_df['Confidence Score'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df.sort_values('Final Score', ascending=False),
                    use_container_width=True
                )

# Footer with additional info
st.markdown("---")
st.markdown("### 📚 About These Visualizations")

with st.expander("ℹ️ Visualization Guide"):
    st.markdown("""
    **Model Performance:**
    - Confusion Matrix: Shows prediction accuracy across categories
    - Classification Report: Detailed precision, recall, and F1-scores
    - Confidence Distribution: Model certainty in predictions
    
    **Feature Analysis:**
    - Keyword Importance: Most influential words for each category
    - Category Overlap: Similarity between job categories
    - Cross-Category Analysis: Common keywords across roles
    
    **Resume Analytics:**
    - Score Breakdown: Components of final resume score
    - Performance Distribution: Statistical analysis of scores
    - Quality Indicators: Word count, keyword matches, etc.
    
    **Interactive Dashboard:**
    - Real-time filtering and analysis
    - Multiple visualization types
    - Dynamic performance metrics
    """)
