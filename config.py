"""
Configuration parameters for TCR Stock Forecasting
"""

# Data Configuration
DATA_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2024-12-31',
    'tickers': ['^GSPC', 'INFY'],  # S&P 500 and Infosys
    'interval': '1d',
}

# Model Configuration
TCR_CONFIG = {
    'max_lags': 10,
    'threshold': 0.001,
    'test_size': 0.2,
}

ARIMA_CONFIG = {
    'order': (5, 1, 2),
}

SARIMA_CONFIG = {
    'order': (1, 1, 1),
    'seasonal_order': (1, 1, 1, 12),
}

SES_CONFIG = {
    'smoothing_level': 0.2,
}

DNM_CONFIG = {
    'neurons': 32,
    'epochs': 50,
}

# Evaluation Metrics
METRICS = ['RMSE', 'MAE', 'MAPE', 'R2']

# Visualization Configuration
PLOT_CONFIG = {
    'figsize': (15, 6),
    'style': 'seaborn-v0_8-darkgrid',
}

# Random Seed
RANDOM_SEED = 42
