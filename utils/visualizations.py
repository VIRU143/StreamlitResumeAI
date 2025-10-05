import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class VisualizationHelper:
    def __init__(self):
        # Modern color palette with vibrant gradients
        self.color_palette = [
            '#667eea', '#764ba2', '#f093fb', '#4facfe',
            '#43e97b', '#fa709a', '#fee140', '#30cfd0',
            '#a8edea', '#fed6e3', '#c471f5', '#fa71cd'
        ]
        self.primary_gradient = ['#667eea', '#764ba2']
        self.secondary_gradient = ['#f093fb', '#4facfe']
    
    def create_confusion_matrix(self, confusion_matrix, class_names):
        """Create an interactive confusion matrix heatmap"""
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=class_names,
            y=class_names,
            color_continuous_scale=[[0, '#e0e7ff'], [0.5, '#818cf8'], [1, '#4f46e5']],
            aspect="auto"
        )
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(confusion_matrix[i][j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i][j] > confusion_matrix.max()/2 else "black")
                )
        
        fig.update_layout(
            title="Confusion Matrix - Model Performance",
            xaxis_title="Predicted Category",
            yaxis_title="Actual Category",
            height=600
        )
        
        return fig
    
    def create_classification_report_chart(self, classification_report):
        """Create a bar chart from classification report"""
        # Extract metrics for each class
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                classes.append(class_name)
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1_score.append(metrics['f1-score'])
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Precision', 'Recall', 'F1-Score'),
            shared_yaxis=True
        )
        
        # Add bars for each metric with gradient colors
        fig.add_trace(
            go.Bar(x=classes, y=precision, name='Precision', marker_color='#667eea'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=classes, y=recall, name='Recall', marker_color='#764ba2'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=classes, y=f1_score, name='F1-Score', marker_color='#f093fb'),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Classification Performance Metrics by Category",
            showlegend=False,
            height=500
        )
        
        # Update y-axis to show values from 0 to 1
        fig.update_yaxes(range=[0, 1])
        
        return fig
    
    def create_class_distribution(self, y_true, y_pred, class_names):
        """Create a comparison of actual vs predicted class distributions"""
        # Count actual and predicted classes
        actual_counts = pd.Series(y_true).value_counts()
        predicted_counts = pd.Series(y_pred).value_counts()
        
        # Ensure all classes are represented
        for class_name in class_names:
            if class_name not in actual_counts:
                actual_counts[class_name] = 0
            if class_name not in predicted_counts:
                predicted_counts[class_name] = 0
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Category': class_names,
            'Actual': [actual_counts.get(cat, 0) for cat in class_names],
            'Predicted': [predicted_counts.get(cat, 0) for cat in class_names]
        })
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Actual',
            x=df['Category'],
            y=df['Actual'],
            marker_color='#667eea',
            marker=dict(line=dict(width=0))
        ))
        
        fig.add_trace(go.Bar(
            name='Predicted',
            x=df['Category'],
            y=df['Predicted'],
            marker_color='#764ba2',
            marker=dict(line=dict(width=0))
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Class Distribution',
            xaxis_title='Category',
            yaxis_title='Count',
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_confidence_distribution(self, y_pred_proba, class_names):
        """Create confidence score distribution"""
        max_confidence = np.max(y_pred_proba, axis=1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=max_confidence,
            nbinsx=20,
            name='Confidence Distribution',
            marker_color='#667eea',
            opacity=0.8,
            marker=dict(line=dict(width=0))
        ))
        
        fig.update_layout(
            title='Model Confidence Score Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, top_features_per_class, selected_class=None):
        """Create feature importance chart for selected class"""
        if not top_features_per_class:
            return None
        
        if selected_class and selected_class in top_features_per_class:
            features = top_features_per_class[selected_class][:15]  # Top 15 features
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=features[::-1],  # Reverse for better readability
                x=list(range(len(features), 0, -1)),  # Importance ranking
                orientation='h',
                marker_color='#667eea',
                marker=dict(line=dict(width=0)),
                name='Feature Importance'
            ))
            
            fig.update_layout(
                title=f'Top Keywords for {selected_class}',
                xaxis_title='Importance Rank',
                yaxis_title='Keywords',
                height=500
            )
            
            return fig
        
        return None
    
    def create_resume_score_breakdown(self, score_data):
        """Create a breakdown of resume score components"""
        categories = ['Confidence', 'Content Richness', 'Keyword Match', 'Final Score']
        values = [
            score_data['confidence_score'],
            score_data['richness_score'],
            score_data['keyword_match_score'],
            score_data['final_score']
        ]
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=categories,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        for i, (category, value, color) in enumerate(zip(categories, values, colors)):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value * 100,  # Convert to percentage
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': color},
                    'bar': {'color': color, 'thickness': 0.75},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(102, 126, 234, 0.1)"},
                        {'range': [50, 80], 'color': "rgba(102, 126, 234, 0.2)"},
                        {'range': [80, 100], 'color': "rgba(102, 126, 234, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 3},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                number={'font': {'size': 32, 'color': color}}
            ), row=row, col=col)
        
        fig.update_layout(
            height=600,
            title="Resume Score Breakdown"
        )
        
        return fig
    
    def create_batch_results_chart(self, results_df):
        """Create visualization for batch processing results"""
        if results_df.empty:
            return None
        
        # Create scatter plot of scores vs confidence
        fig = px.scatter(
            results_df,
            x='Confidence Score',
            y='Final Score',
            color='Predicted Category',
            size='Word Count',
            hover_data=['Filename'],
            title='Resume Analysis Results - Score vs Confidence'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_category_pie_chart(self, results_df):
        """Create pie chart of predicted categories"""
        if results_df.empty:
            return None
        
        category_counts = results_df['Predicted Category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Distribution of Predicted Categories'
        )
        
        fig.update_layout(height=400)
        
        return fig
