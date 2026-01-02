"""
Citation Impact Prediction - Streamlit App
AUB Capstone Project

Predicts citation impact for academic publications using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Citation Impact Predictor - AUB",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    models = {}

    try:
        # Try loading tuned models first
        tuned_cls_path = Path('../models/tuned/lightgbm_classifier_tuned.pkl')
        tuned_reg_path = Path('../models/tuned/lightgbm_regressor_tuned.pkl')

        if tuned_cls_path.exists() and tuned_reg_path.exists():
            with open(tuned_cls_path, 'rb') as f:
                models['classifier'] = pickle.load(f)
            with open(tuned_reg_path, 'rb') as f:
                models['regressor'] = pickle.load(f)
            models['model_type'] = 'Tuned (Optimized)'
            st.sidebar.success("✓ Tuned models loaded")
        else:
            # Fall back to baseline models
            baseline_cls_path = Path('../models/saved/lightgbm_classifier.pkl')
            baseline_reg_path = Path('../models/saved/lightgbm_regressor.pkl')

            if baseline_cls_path.exists() and baseline_reg_path.exists():
                with open(baseline_cls_path, 'rb') as f:
                    models['classifier'] = pickle.load(f)
                with open(baseline_reg_path, 'rb') as f:
                    models['regressor'] = pickle.load(f)
                models['model_type'] = 'Baseline'
                st.sidebar.warning("⚠️ Using baseline models (tuned models not found)")
            else:
                st.sidebar.error("❌ No models found! Run model training first.")
                return None

        # Load TF-IDF vectorizer
        tfidf_path = Path('../models/tfidf_vectorizer.pkl')
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                models['tfidf'] = pickle.load(f)
            st.sidebar.success("✓ TF-IDF vectorizer loaded")
        else:
            st.sidebar.error("❌ TF-IDF vectorizer not found!")
            return None

        # Load feature info
        feature_info_path = Path('../models/feature_info.json')
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                models['feature_info'] = json.load(f)
            st.sidebar.success(f"✓ Feature info loaded ({models['feature_info']['total_features']} features)")
        else:
            st.sidebar.warning("⚠️ Feature info not found")

        return models

    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None


def extract_features(abstract, venue_sjr, venue_snip, venue_citescore,
                     num_authors, year, tfidf_vectorizer):
    """
    Extract features from user inputs to match training data format
    """
    current_year = datetime.now().year

    # 1. Temporal features
    years_since_publication = current_year - year
    publication_age_log = np.log1p(years_since_publication)

    # 2. Collaboration features
    def categorize_collaboration(n):
        if n == 1:
            return 'single'
        elif n <= 5:
            return 'small'
        elif n <= 10:
            return 'medium'
        elif n <= 50:
            return 'large'
        else:
            return 'mega'

    collab_size = categorize_collaboration(num_authors)
    is_mega_collab = 1 if num_authors >= 50 else 0

    # One-hot encode collaboration (matching training: drop 'large')
    collab_medium = 1 if collab_size == 'medium' else 0
    collab_mega = 1 if collab_size == 'mega' else 0
    collab_single = 1 if collab_size == 'single' else 0
    collab_small = 1 if collab_size == 'small' else 0

    # 3. TF-IDF features from abstract
    tfidf_features = tfidf_vectorizer.transform([abstract]).toarray()[0]

    # 4. Base features
    base_features = {
        'Number of Authors': num_authors,
        'is_mega_collab': is_mega_collab,
        'years_since_publication': years_since_publication,
        'publication_age_log': publication_age_log,
        'venue_snip': venue_snip,
        'venue_citescore': venue_citescore,
        'venue_sjr': venue_sjr,
    }

    # 5. Collaboration dummy features
    collab_features = {
        'collab_medium': collab_medium,
        'collab_mega': collab_mega,
        'collab_single': collab_single,
        'collab_small': collab_small,
    }

    # 6. Combine all features in correct order
    # Order: base_features, collab_features, tfidf_features
    feature_vector = []

    # Add base features
    for key in ['Number of Authors', 'is_mega_collab', 'years_since_publication',
                'publication_age_log', 'venue_snip', 'venue_citescore', 'venue_sjr']:
        feature_vector.append(base_features[key])

    # Add collaboration features
    for key in ['collab_medium', 'collab_mega', 'collab_single', 'collab_small']:
        feature_vector.append(collab_features[key])

    # Add TF-IDF features
    feature_vector.extend(tfidf_features)

    return np.array(feature_vector).reshape(1, -1)


def predict_impact(models, features):
    """Make predictions using loaded models"""
    predictions = {}

    # Classification prediction
    if models['classifier']:
        pred_class = models['classifier'].predict(features)[0]
        pred_proba = models['classifier'].predict_proba(features)[0]

        predictions['high_impact'] = bool(pred_class)
        predictions['high_impact_probability'] = float(pred_proba[1])
        predictions['standard_probability'] = float(pred_proba[0])

    # Regression prediction
    if models['regressor']:
        pred_citations_log = models['regressor'].predict(features)[0]

        # Convert from log space to real citations
        pred_citations = np.expm1(pred_citations_log)  # inverse of log1p

        # Confidence interval (±1 std in log space ≈ 2x in real space)
        confidence_multiplier = np.exp(0.9)  # Based on RMSE ~0.9
        lower_bound = pred_citations / confidence_multiplier
        upper_bound = pred_citations * confidence_multiplier

        predictions['citations_log'] = float(pred_citations_log)
        predictions['citations'] = float(pred_citations)
        predictions['citations_lower'] = float(lower_bound)
        predictions['citations_upper'] = float(upper_bound)

    return predictions


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<p class="main-header">📚 Citation Impact Prediction System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">American University of Beirut - Capstone Project</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    models = load_models()

    if models is None:
        st.error("⚠️ **Models not loaded!** Please ensure model files exist in `models/` directory.")
        st.info("""
        **Required files:**
        - `models/tuned/lightgbm_classifier_tuned.pkl` (or `models/saved/lightgbm_classifier.pkl`)
        - `models/tuned/lightgbm_regressor_tuned.pkl` (or `models/saved/lightgbm_regressor.pkl`)
        - `models/tfidf_vectorizer.pkl`
        - `models/feature_info.json`
        """)
        return

    # Sidebar - Model Info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.metric("Model Type", models['model_type'])

        if models['model_type'] == 'Tuned (Optimized)':
            st.success("Using hyperparameter-optimized models")
            st.write("**Performance:**")
            st.write("- Classification F1: 63.3%")
            st.write("- Regression R²: 58.3%")
        else:
            st.write("**Performance:**")
            st.write("- Classification F1: 62.8%")
            st.write("- Regression R²: 57.3%")

        st.markdown("---")
        st.info("""
        **How it works:**
        1. Enter paper details
        2. Model extracts 311 features
        3. Predicts citation impact
        """)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Paper Prediction", "📊 Batch Prediction", "📈 Model Insights"])

    # ========================================
    # TAB 1: Single Paper Prediction
    # ========================================
    with tab1:
        st.header("Predict Citation Impact for a Single Paper")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Paper Details")

            # Abstract input
            abstract = st.text_area(
                "📄 Abstract",
                height=200,
                placeholder="Paste the paper abstract here...",
                help="The abstract text will be processed using TF-IDF (300 features)"
            )

            # Publication year
            year = st.number_input(
                "📅 Publication Year",
                min_value=2000,
                max_value=datetime.now().year,
                value=2020,
                help="Year the paper was published"
            )

            # Number of authors
            num_authors = st.number_input(
                "👥 Number of Authors",
                min_value=1,
                max_value=1000,
                value=5,
                help="Total number of authors on the paper"
            )

        with col2:
            st.subheader("Venue Metrics")
            st.caption("Journal/conference prestige indicators")

            venue_sjr = st.number_input(
                "📊 SJR (SCImago Journal Rank)",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.1,
                help="Most important feature! Typical range: 0.1-5.0"
            )

            venue_snip = st.number_input(
                "📈 SNIP",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Source Normalized Impact per Paper"
            )

            venue_citescore = st.number_input(
                "⭐ CiteScore",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5,
                help="Average citations per paper"
            )

        # Predict button
        st.markdown("---")
        if st.button("🔮 Predict Citation Impact", type="primary", use_container_width=True):
            if not abstract or len(abstract) < 50:
                st.error("⚠️ Please enter an abstract (at least 50 characters)")
            else:
                with st.spinner("Extracting features and making predictions..."):
                    # Extract features
                    features = extract_features(
                        abstract, venue_sjr, venue_snip, venue_citescore,
                        num_authors, year, models['tfidf']
                    )

                    # Make predictions
                    predictions = predict_impact(models, features)

                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Results")

                    # High-Impact Classification
                    st.markdown("### 📊 High-Impact Classification")

                    prob_high = predictions['high_impact_probability'] * 100
                    prob_std = predictions['standard_probability'] * 100

                    if predictions['high_impact']:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>✨ HIGH-IMPACT PAPER</h2>
                            <h1>{prob_high:.1f}%</h1>
                            <p>Probability of being in top 25% (≥26 citations)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📄 Standard Paper</h3>
                            <h2>{prob_std:.1f}%</h2>
                            <p>Probability of being in bottom 75%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence meter
                    fig_prob = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_high,
                        title={'text': "High-Impact Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "lightyellow"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "lightblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_prob.update_layout(height=300)
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # Citation Count Prediction
                    st.markdown("### 📈 Expected Citation Count")

                    pred_cit = predictions['citations']
                    lower = predictions['citations_lower']
                    upper = predictions['citations_upper']

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Lower Bound", f"{lower:.0f}")
                    with col_b:
                        st.metric("Predicted", f"{pred_cit:.0f}", help="Expected citations")
                    with col_c:
                        st.metric("Upper Bound", f"{upper:.0f}")

                    st.info(f"**Confidence Interval:** {lower:.0f} - {upper:.0f} citations (68% confidence)")

                    # Paper age warning
                    current_year = datetime.now().year
                    age = current_year - year

                    if age < 3:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>⚠️ Recent Paper Warning</strong><br>
                            This paper is only {age} years old. Citation predictions are less reliable
                            for recent papers (accuracy drops ~30% for papers <3 years old).
                            Actual citations may be significantly lower than predicted.
                        </div>
                        """, unsafe_allow_html=True)

                    # Key factors
                    st.markdown("### 🔍 Key Predictive Factors")
                    st.write("""
                    Based on feature importance analysis, the main factors influencing this prediction are:
                    """)

                    factors = pd.DataFrame({
                        'Factor': ['Venue SJR', 'Publication Age', 'Venue Metrics', 'Author Count', 'Abstract Content'],
                        'Importance': [32, 24, 16, 5, 23],
                        'Your Value': [
                            f"{venue_sjr:.2f}",
                            f"{age} years",
                            f"SNIP={venue_snip:.1f}, CS={venue_citescore:.1f}",
                            f"{num_authors} authors",
                            f"{len(abstract)} chars"
                        ]
                    })

                    fig_factors = px.bar(
                        factors,
                        x='Importance',
                        y='Factor',
                        orientation='h',
                        text='Your Value',
                        title='Feature Importance (%)'
                    )
                    fig_factors.update_traces(textposition='outside')
                    fig_factors.update_layout(height=400)
                    st.plotly_chart(fig_factors, use_container_width=True)

    # ========================================
    # TAB 2: Batch Prediction
    # ========================================
    with tab2:
        st.header("Batch Prediction from CSV")
        st.write("Upload a CSV file with multiple papers to get predictions for all of them.")

        st.info("""
        **Required columns:**
        - `Abstract`: Paper abstract text
        - `Year`: Publication year
        - `Number of Authors`: Author count
        - `SJR`: SCImago Journal Rank
        - `SNIP`: Source Normalized Impact per Paper
        - `CiteScore`: CiteScore metric
        """)

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} papers")
                st.dataframe(df.head())

                if st.button("Process All Papers"):
                    with st.spinner("Processing papers..."):
                        results = []

                        for idx, row in df.iterrows():
                            try:
                                features = extract_features(
                                    row['Abstract'],
                                    row.get('SJR', 1.0),
                                    row.get('SNIP', 1.0),
                                    row.get('CiteScore', 3.0),
                                    row.get('Number of Authors', 5),
                                    row['Year'],
                                    models['tfidf']
                                )

                                preds = predict_impact(models, features)

                                results.append({
                                    'Index': idx,
                                    'High_Impact_Probability': preds['high_impact_probability'],
                                    'Predicted_Citations': preds['citations'],
                                    'Citations_Lower': preds['citations_lower'],
                                    'Citations_Upper': preds['citations_upper']
                                })
                            except Exception as e:
                                st.warning(f"Error processing row {idx}: {e}")

                        results_df = pd.DataFrame(results)
                        results_df = pd.concat([df, results_df.drop('Index', axis=1)], axis=1)

                        st.success(f"✓ Processed {len(results_df)} papers")
                        st.dataframe(results_df)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # ========================================
    # TAB 3: Model Insights
    # ========================================
    with tab3:
        st.header("Model Performance & Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Classification Performance")

            if models['model_type'] == 'Tuned (Optimized)':
                metrics_cls = {
                    'Accuracy': 77.8,
                    'Precision': 54.4,
                    'Recall': 75.7,
                    'F1-Score': 63.3,
                    'AUC-ROC': 84.9
                }
            else:
                metrics_cls = {
                    'Accuracy': 77.6,
                    'Precision': 54.1,
                    'Recall': 74.7,
                    'F1-Score': 62.8,
                    'AUC-ROC': 85.1
                }

            for metric, value in metrics_cls.items():
                st.metric(metric, f"{value}%")

        with col2:
            st.subheader("📈 Regression Performance")

            if models['model_type'] == 'Tuned (Optimized)':
                st.metric("R² Score", "58.3%", help="Explains 58% of citation variance")
                st.metric("RMSE", "0.906", help="Root Mean Squared Error (log-space)")
                st.metric("MAE", "0.714", help="Mean Absolute Error (log-space)")
            else:
                st.metric("R² Score", "57.3%", help="Explains 57% of citation variance")
                st.metric("RMSE", "0.917", help="Root Mean Squared Error (log-space)")
                st.metric("MAE", "0.721", help="Mean Absolute Error (log-space)")

        st.markdown("---")
        st.subheader("🎯 Top Predictive Features")

        feature_importance = pd.DataFrame({
            'Feature': [
                'Venue SJR',
                'Publication Age (log)',
                'Years Since Publication',
                'Venue SNIP',
                'Number of Authors',
                'Venue CiteScore',
                'Abstract Content (TF-IDF)',
                'Collaboration Size',
                'Is Mega-Collaboration'
            ],
            'Importance (%)': [32.0, 12.7, 11.1, 5.2, 4.8, 4.3, 23.0, 4.0, 0.8]
        })

        fig = px.bar(
            feature_importance.sort_values('Importance (%)', ascending=True),
            x='Importance (%)',
            y='Feature',
            orientation='h',
            title='Feature Importance for Citation Prediction'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("ℹ️ About This Model")
        st.write(f"""
        **Model Architecture:** LightGBM (Gradient Boosting)

        **Training Data:** 15,471 academic papers (2010-2025)

        **Feature Count:** 311 features
        - 7 base features (authors, temporal, venue)
        - 4 collaboration categories
        - 300 TF-IDF text features from abstracts

        **Performance Notes:**
        - Best for papers ≥3 years old (accuracy ~63% F1)
        - Less reliable for recent papers <3 years (accuracy ~33% F1)
        - Venue prestige (SJR) is the strongest single predictor (32% importance)
        - Temporal features critical (publication age accounts for 24% of predictions)

        **Limitations:**
        - Cannot predict breakthrough discoveries (randomness in citations)
        - Field-specific citation cultures not fully captured
        - Self-citations not distinguished
        - Negative citations (retractions) not considered
        """)


if __name__ == "__main__":
    main()
