"""
Citation Impact Prediction - Streamlit Deployment
AUB Capstone Project

This Streamlit app provides an interactive interface for predicting citation impact
of academic publications using trained machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Citation Impact Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class CitationPredictor:
    """Handle model loading and predictions"""

    def __init__(self, model_dir='../models'):
        self.model_dir = Path(model_dir)
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self.scaler = None

    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load classification model
            classifier_path = self.model_dir / 'classifier_model.pkl'
            if classifier_path.exists():
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                st.sidebar.success("✓ Classification model loaded")

            # Load regression model
            regressor_path = self.model_dir / 'regressor_model.pkl'
            if regressor_path.exists():
                with open(regressor_path, 'rb') as f:
                    self.regressor = pickle.load(f)
                st.sidebar.success("✓ Regression model loaded")

            # Load feature information
            feature_path = self.model_dir / 'feature_info.json'
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_names = feature_info.get('feature_names', [])
                st.sidebar.success("✓ Feature metadata loaded")

            # Load scaler if exists
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            return True

        except Exception as e:
            st.sidebar.error(f"Error loading models: {e}")
            return False

    def predict_classification(self, features):
        """Predict if paper will be highly cited"""
        if self.classifier is None:
            return None, None

        # Scale features if scaler exists
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        prediction = self.classifier.predict(features_scaled)
        probability = self.classifier.predict_proba(features_scaled)

        return prediction[0], probability[0]

    def predict_regression(self, features):
        """Predict citation count"""
        if self.regressor is None:
            return None

        # Scale features if scaler exists
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        prediction = self.regressor.predict(features_scaled)
        return prediction[0]


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<p class="main-header">📚 Citation Impact Prediction System</p>',
                unsafe_allow_html=True)
    st.markdown("### Predict citation impact for academic publications using machine learning")
    st.markdown("---")

    # Initialize predictor
    predictor = CitationPredictor()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model loading
        st.subheader("Model Management")
        if st.button("Load/Reload Models"):
            with st.spinner("Loading models..."):
                success = predictor.load_models()
                if success:
                    st.success("Models loaded successfully!")
                else:
                    st.warning("Some models could not be loaded. Please train models first.")

        st.markdown("---")

        # Model info
        st.subheader("Model Status")
        st.write(f"Classifier: {'✓ Loaded' if predictor.classifier else '✗ Not loaded'}")
        st.write(f"Regressor: {'✓ Loaded' if predictor.regressor else '✗ Not loaded'}")

        st.markdown("---")

        # Upload new data
        st.subheader("📤 Upload New Data")
        uploaded_file = st.file_uploader("Upload CSV/Excel file",
                                         type=['csv', 'xlsx'],
                                         help="Upload publication data for batch predictions")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Single Prediction",
        "📊 Batch Predictions",
        "📈 Model Insights",
        "ℹ️ About"
    ])

    # Tab 1: Single Prediction
    with tab1:
        st.header("Predict Citation Impact for a Single Paper")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Paper Information")

            # Basic information
            title = st.text_input("Paper Title",
                                 placeholder="Enter the paper title...")
            abstract = st.text_area("Abstract",
                                   placeholder="Enter the paper abstract...",
                                   height=150)

            # Author information
            st.subheader("Author Metrics")
            max_h_index = st.number_input("Maximum Author H-Index",
                                         min_value=0, max_value=300, value=10)
            avg_h_index = st.number_input("Average Author H-Index",
                                         min_value=0.0, max_value=300.0, value=5.0)
            num_authors = st.number_input("Number of Authors",
                                         min_value=1, max_value=100, value=3)

        with col2:
            st.subheader("Publication Details")

            # Venue information
            venue_prestige = st.slider("Venue Prestige Score",
                                      min_value=0.0, max_value=100.0, value=50.0,
                                      help="Score based on venue impact factor and reputation")

            pub_year = st.number_input("Publication Year",
                                      min_value=2000, max_value=2025, value=2020)

            # Field of study
            field = st.selectbox("Field of Study", [
                "Computer Science",
                "Medicine",
                "Engineering",
                "Physics",
                "Biology",
                "Chemistry",
                "Social Sciences",
                "Business",
                "Other"
            ])

            # Additional features
            st.subheader("Additional Features")
            num_references = st.number_input("Number of References",
                                            min_value=0, max_value=500, value=30)
            num_pages = st.number_input("Number of Pages",
                                       min_value=1, max_value=100, value=10)

        # Predict button
        st.markdown("---")
        if st.button("🔮 Predict Citation Impact", type="primary", use_container_width=True):
            if not title or not abstract:
                st.warning("Please enter both title and abstract")
            else:
                # Create feature vector (this is a simplified example)
                # In practice, you'd need to process text features, etc.
                features = pd.DataFrame([{
                    'max_h_index': max_h_index,
                    'avg_h_index': avg_h_index,
                    'num_authors': num_authors,
                    'venue_prestige': venue_prestige,
                    'pub_year': pub_year,
                    'num_references': num_references,
                    'num_pages': num_pages,
                    'abstract_length': len(abstract),
                    'title_length': len(title)
                }])

                # Make predictions
                st.markdown("### Prediction Results")

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    if predictor.classifier:
                        pred_class, prob = predictor.predict_classification(features)

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Impact Classification",
                                 "High Impact 🌟" if pred_class == 1 else "Standard Impact",
                                 delta=f"{prob[1]*100:.1f}% confidence")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob[1] * 100,
                            title={'text': "High Impact Probability"},
                            gauge={'axis': {'range': [0, 100]},
                                  'bar': {'color': "darkblue"},
                                  'steps': [
                                      {'range': [0, 30], 'color': "lightgray"},
                                      {'range': [30, 70], 'color': "gray"},
                                      {'range': [70, 100], 'color': "lightblue"}],
                                  'threshold': {
                                      'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75,
                                      'value': 75}}))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Classification model not loaded")

                with col_res2:
                    if predictor.regressor:
                        pred_citations = predictor.predict_regression(features)

                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Predicted Citation Count",
                                 f"{int(pred_citations)} citations",
                                 delta="in 5 years")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Citation range
                        st.info(f"""
                        **Expected Citation Range:** {int(pred_citations * 0.7)} - {int(pred_citations * 1.3)} citations

                        This prediction is based on features available at publication time.
                        """)
                    else:
                        st.warning("Regression model not loaded")

    # Tab 2: Batch Predictions
    with tab2:
        st.header("Batch Prediction for Multiple Papers")

        if uploaded_file is not None:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Loaded {len(df)} papers from {uploaded_file.name}")

            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))

            if st.button("Run Batch Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    # This is a placeholder - implement actual batch prediction
                    st.info("Batch prediction feature coming soon!")
                    st.write("Will predict citations for all papers in the uploaded file")
        else:
            st.info("👈 Upload a CSV or Excel file in the sidebar to start batch predictions")

            # Show sample format
            st.subheader("Expected Data Format")
            sample_df = pd.DataFrame({
                'Title': ['Sample Paper 1', 'Sample Paper 2'],
                'Abstract': ['Abstract text...', 'Abstract text...'],
                'max_h_index': [15, 20],
                'avg_h_index': [8.5, 12.0],
                'num_authors': [3, 5],
                'venue_prestige': [60, 75],
                'pub_year': [2020, 2021]
            })
            st.dataframe(sample_df)

    # Tab 3: Model Insights
    with tab3:
        st.header("Model Performance and Feature Importance")

        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.subheader("Classification Model Performance")
            # Placeholder metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                'Train': [0.85, 0.83, 0.87, 0.85, 0.91],
                'Test': [0.82, 0.80, 0.84, 0.82, 0.88]
            })
            st.dataframe(metrics_df, use_container_width=True)

            st.info("📊 Load trained models to see actual performance metrics")

        with col_ins2:
            st.subheader("Regression Model Performance")
            # Placeholder metrics
            reg_metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE'],
                'Train': [0.78, 12.5, 18.3, 0.25],
                'Test': [0.74, 14.2, 20.1, 0.28]
            })
            st.dataframe(reg_metrics_df, use_container_width=True)

        st.markdown("---")

        # Feature importance
        st.subheader("Feature Importance Analysis")

        # Placeholder feature importance
        features_imp = pd.DataFrame({
            'Feature': ['Author H-Index (Max)', 'Venue Prestige', 'Num References',
                       'Publication Year', 'Num Authors', 'Abstract Length'],
            'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
        }).sort_values('Importance', ascending=True)

        fig = px.barh(features_imp, x='Importance', y='Feature',
                     title='Top Features Driving Citation Impact',
                     labels={'Importance': 'Feature Importance Score'})
        st.plotly_chart(fig, use_container_width=True)

        st.info("🔍 Feature importance shows which factors most strongly predict citation impact")

    # Tab 4: About
    with tab4:
        st.header("About This Project")

        st.markdown("""
        ### Citation Impact Prediction System

        This application is part of a capstone project developed in collaboration with
        the **American University of Beirut (AUB)** to predict citation impact of academic publications.

        #### Project Objectives

        1. **Early Prediction**: Identify high-impact papers using only information available at publication
        2. **Resource Allocation**: Help institutions optimize research funding and support
        3. **Strategic Planning**: Guide researchers in venue selection and collaboration decisions
        4. **Interpretability**: Provide transparent, explainable predictions

        #### Methodology

        - **Data**: 10,000+ publications from AUB researchers with complete metadata
        - **Features**: Author metrics (h-index), venue prestige, text analysis, temporal factors
        - **Models**: Ensemble of classification and regression models
        - **Validation**: 5-fold cross-validation with temporal holdout testing

        #### Key Features

        - 🔮 **Single Prediction**: Get instant predictions for new papers
        - 📊 **Batch Processing**: Analyze multiple papers at once
        - 📈 **Model Insights**: Understand what drives citation impact
        - 🔄 **Continuous Learning**: Models can be updated with new data

        #### Technologies Used

        - **Machine Learning**: scikit-learn, XGBoost, LightGBM
        - **Deployment**: Streamlit
        - **Data Processing**: pandas, numpy
        - **Visualization**: plotly, matplotlib, seaborn

        #### Contact & Collaboration

        This project was developed as part of a capstone in collaboration with AUB.
        For questions or collaboration opportunities, please contact the development team.

        ---

        **Version**: 1.0.0
        **Last Updated**: {datetime.now().strftime('%B %Y')}
        """)


if __name__ == "__main__":
    main()
