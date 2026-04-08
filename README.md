# AutoML Time Series Forecasting with AutoGluon

## Overview

This Jupyter notebook demonstrates automated time series forecasting using **AutoGluon's TimeSeriesPredictor** on the M4 Hourly dataset. The notebook explores various model configurations, hyperparameter tuning, and evaluation techniques for time series prediction.

## Key Features

- **Dataset**: M4 Hourly competition dataset loaded via `gluonts`
- **AutoML Framework**: Amazon's AutoGluon TimeSeriesPredictor
- **Prediction Horizon**: 24 time steps ahead
- **Evaluation Metric**: MASE (Mean Absolute Scaled Error)

## Notebook Structure

### 1. Data Loading & Preprocessing
- Loads the M4 hourly dataset using gluonts
- Converts dataset to pandas DataFrame format with proper timestamp handling
- Processes data into `TimeSeriesDataFrame` format required by AutoGluon
- Saves processed data as pickle files for reuse

### 2. Initial Model Training
- Trains multiple time series models including:
  - Naive
  - SeasonalNaive
  - RecursiveTabular
  - DirectTabular
- Weighted ensemble combines the best models
- Time limit: 500 seconds

### 3. Hyperparameter Tuning
- Experiments with different model configurations:
  - DeepAR (probabilistic forecasting with RNNs)
  - ETS (Exponential Smoothing) with additive seasonality
  - Custom hyperparameter spaces using `space` module
- Compares performance across different model combinations

### 4. Visualization
- Time series plots showing:
  - Training data
  - Forecasted means
  - Test set actual values
  - 10%-90% confidence intervals

### 5. Data Subsampling
- Demonstrates working with subsampled data (every 10th observation)
- Compares model performance on different data densities

## Key Code Sections

### Data Processing Function
```python
def process_dataset(dataset, metadata_freq):
    # Converts gluonts dataset to pandas DataFrame
    # Creates date range with proper frequency
    # Returns DataFrame with date, value, and item_id columns
```

### Model Configuration
```python
predictor = TimeSeriesPredictor(
    prediction_length=24,
    path="autogluon-m4-monthly",
    target="value",
    eval_metric="MASE",
)
```

### Hyperparameter Tuning Example
```python
hyperparameters={
    "DeepAR": {},
    "ETS": [
        {"seasonal": "add"},
        {"seasonal": None},
    ],
}
```

## Results

The notebook compares multiple model configurations:
- **Initial ensemble**: Weighted combination of Naive, SeasonalNaive, RecursiveTabular, and DirectTabular
- **DeepAR + ETS combination**: Achieves different validation scores with alternative model selection

## Dependencies

```
autogluon.timeseries
gluonts
pandas
matplotlib
```

## Usage Notes

1. **Data Requirements**: Time series must have sufficient length for validation windows
2. **Memory Considerations**: The full M4 hourly dataset is large (353k+ rows), subsampling may be necessary for faster iteration
3. **Model Selection**: The predictor automatically selects the best model based on validation scores
4. **Time Limits**: Training time can be adjusted via the `time_limit` parameter

## Key Insights

- Weighted ensembles typically outperform individual models
- DeepAR shows competitive performance for probabilistic forecasting
- Data subsampling can significantly reduce training time with minimal performance impact
- Proper timestamp handling is critical for accurate forecasting

## Error Handling

The notebook addresses common issues:
- Short time series are automatically filtered out (minimum length requirements)
- Path conflicts are warned when overwriting existing predictors
- Missing validation windows are handled with appropriate warnings
