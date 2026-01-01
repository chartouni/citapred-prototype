# Citation Impact Predictor - Streamlit App

This directory contains the Streamlit web application for deploying citation impact prediction models.

## Running the App

### Prerequisites

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Ensure you have trained models in the `../models/` directory:
   - `classifier_model.pkl` - Classification model
   - `regressor_model.pkl` - Regression model
   - `feature_info.json` - Feature metadata
   - `scaler.pkl` - Feature scaler (optional)

### Launch the App

```bash
cd app
streamlit run streamlit_app.py
```

Or from the project root:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at: **http://localhost:8501**

## Features

### 1. Single Prediction
Predict citation impact for a single paper by entering:
- Paper title and abstract
- Author metrics (h-index, number of authors)
- Venue prestige score
- Publication details

Get:
- High/Standard impact classification with confidence
- Predicted citation count
- Probability gauge visualization

### 2. Batch Predictions
Upload a CSV or Excel file containing multiple papers to:
- Get predictions for all papers at once
- Download results as CSV
- View summary statistics

### 3. Model Insights
View:
- Classification and regression model performance
- Feature importance rankings
- Interactive visualizations

### 4. Model Management
- Load/reload models dynamically
- Check model status
- Update models without restarting the app

## File Structure

```
app/
├── streamlit_app.py    # Main Streamlit application
├── utils.py            # Utility functions for predictions
└── README.md           # This file
```

## Expected Model Format

Models should be saved using pickle:

```python
import pickle

# Save classifier
with open('../models/classifier_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save regressor
with open('../models/regressor_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Save feature metadata
import json
feature_info = {
    'feature_names': ['max_h_index', 'venue_prestige', ...],
    'feature_types': {'max_h_index': 'numeric', ...}
}
with open('../models/feature_info.json', 'w') as f:
    json.dump(feature_info, f)
```

## Customization

### Adding New Features

1. Update the input form in `streamlit_app.py` (Tab 1)
2. Add feature extraction logic in `utils.py`
3. Update the feature vector creation
4. Retrain models with new features

### Modifying UI

The app uses:
- Custom CSS for styling (see `st.markdown` sections)
- Plotly for interactive charts
- Streamlit columns for layout

### Deployment Options

#### Local Deployment
Already configured - just run `streamlit run streamlit_app.py`

#### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set up secrets for any sensitive data
4. Deploy!

#### Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

## Troubleshooting

### Models Not Loading
- Check that model files exist in `../models/`
- Verify file permissions
- Check pickle compatibility (same Python version)

### Slow Predictions
- Check if scaler is needed but not provided
- Optimize feature extraction in `utils.py`
- Consider caching predictions with `@st.cache_data`

### Port Already in Use
Run on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## Future Enhancements

- [ ] Add SHAP explanations for predictions
- [ ] Implement model comparison (A/B testing)
- [ ] Add user authentication
- [ ] Create API endpoints for programmatic access
- [ ] Add model retraining workflow
- [ ] Implement prediction history tracking

## Support

For issues or questions, refer to:
- Main project README
- PROJECT_ROADMAP.md
- Streamlit documentation: https://docs.streamlit.io
