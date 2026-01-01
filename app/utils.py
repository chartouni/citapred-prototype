"""
Utility functions for citation prediction
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_features(text_series, max_features=100):
    """
    Extract TF-IDF features from text

    Args:
        text_series: pandas Series containing text
        max_features: maximum number of TF-IDF features

    Returns:
        DataFrame with TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(text_series.fillna(''))

    feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=text_series.index
    )

    return tfidf_df, vectorizer


def extract_author_features(author_data):
    """
    Extract features from author information

    Args:
        author_data: DataFrame or Series containing author information

    Returns:
        DataFrame with author features
    """
    features = pd.DataFrame()

    # Placeholder - adapt based on your actual data structure
    if isinstance(author_data, pd.Series):
        # Extract from series
        features['num_authors'] = author_data.str.count(';') + 1

    return features


def calculate_venue_prestige(venue_name, venue_lookup):
    """
    Calculate venue prestige score

    Args:
        venue_name: Name of publication venue
        venue_lookup: Dictionary mapping venue names to prestige scores

    Returns:
        Prestige score (0-100)
    """
    if venue_name in venue_lookup:
        return venue_lookup[venue_name]
    else:
        return 50.0  # Default/unknown venue score


def create_feature_vector(paper_data, include_text=True):
    """
    Create feature vector for a single paper

    Args:
        paper_data: Dictionary containing paper information
        include_text: Whether to include text features

    Returns:
        pandas DataFrame with features
    """
    features = {}

    # Basic numeric features
    numeric_fields = [
        'max_h_index', 'avg_h_index', 'num_authors',
        'venue_prestige', 'pub_year', 'num_references', 'num_pages'
    ]

    for field in numeric_fields:
        features[field] = paper_data.get(field, 0)

    # Text length features
    if 'abstract' in paper_data:
        features['abstract_length'] = len(paper_data['abstract'])
        features['abstract_word_count'] = len(paper_data['abstract'].split())

    if 'title' in paper_data:
        features['title_length'] = len(paper_data['title'])
        features['title_word_count'] = len(paper_data['title'].split())

    return pd.DataFrame([features])


def clean_text(text):
    """
    Clean and preprocess text

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Convert to lowercase
    text = text.lower()

    return text


def calculate_citation_category(citation_count, percentile_threshold=75):
    """
    Categorize papers as high-impact or standard based on citation count

    Args:
        citation_count: Number of citations
        percentile_threshold: Percentile threshold for high-impact (default 75)

    Returns:
        Binary category (1=high impact, 0=standard)
    """
    # This is a simplified version - in practice, calculate from your dataset
    if citation_count >= percentile_threshold:
        return 1
    else:
        return 0


def prepare_features_for_prediction(df, text_columns=['Abstract', 'Title']):
    """
    Prepare features for prediction from raw dataframe

    Args:
        df: DataFrame with raw paper data
        text_columns: List of text columns to process

    Returns:
        DataFrame with processed features
    """
    features_df = df.copy()

    # Clean text columns
    for col in text_columns:
        if col in features_df.columns:
            features_df[f'{col}_clean'] = features_df[col].apply(clean_text)
            features_df[f'{col}_length'] = features_df[col].fillna('').str.len()

    return features_df


def get_model_metrics(y_true, y_pred, y_prob=None, task='classification'):
    """
    Calculate model performance metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for classification)
        task: 'classification' or 'regression'

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
    )

    metrics = {}

    if task == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')

        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)

    elif task == 'regression':
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['mape'] = mape

    return metrics
