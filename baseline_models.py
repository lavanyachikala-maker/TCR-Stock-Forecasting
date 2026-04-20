"""
Baseline models for stock forecasting comparison
- ARIMA
- SARIMA
- Single Exponential Smoothing (SES)
- Dendritic Neuron Model (DNM)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA Model for time series forecasting"""
    
    def __init__(self, order=(5, 1, 2), verbose=True):
        """
        Parameters:
        -----------
        order : tuple
            ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.verbose = verbose
    
    def fit(self, data):
        """Fit ARIMA model"""
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        
        if self.verbose:
            print(f"Fitting ARIMA{self.order}...")
        
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        
        return self
    
    def predict(self, steps=1):
        """Make predictions"""
        forecast = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast.predicted_mean.values
        return predictions
    
    def get_info(self):
        return {
            'model_type': 'ARIMA',
            'order': self.order,
        }


class SARIMAModel:
    """SARIMA Model for seasonal time series forecasting"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), verbose=True):
        """
        Parameters:
        -----------
        order : tuple
            ARIMA order (p, d, q)
        seasonal_order : tuple
            Seasonal order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.verbose = verbose
    
    def fit(self, data):
        """Fit SARIMA model"""
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        
        if self.verbose:
            print(f"Fitting SARIMA{self.order}x{self.seasonal_order}...")
        
        self.model = SARIMAX(data, 
                            order=self.order, 
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        self.fitted_model = self.model.fit(disp=False)
        
        return self
    
    def predict(self, steps=1):
        """Make predictions"""
        forecast = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast.predicted_mean.values
        return predictions
    
    def get_info(self):
        return {
            'model_type': 'SARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
        }


class SESModel:
    """Single Exponential Smoothing Model"""
    
    def __init__(self, smoothing_level=0.2, verbose=True):
        """
        Parameters:
        -----------
        smoothing_level : float
            Smoothing level (alpha)
        """
        self.smoothing_level = smoothing_level
        self.model = None
        self.fitted_model = None
        self.verbose = verbose
    
    def fit(self, data):
        """Fit SES model"""
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        
        if self.verbose:
            print(f"Fitting SES (alpha={self.smoothing_level})...")
        
        self.model = ExponentialSmoothing(data, trend=None, seasonal=None)
        self.fitted_model = self.model.fit(smoothing_level=self.smoothing_level)
        
        return self
    
    def predict(self, steps=1):
        """Make predictions"""
        forecast = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast.predicted_mean.values
        return predictions
    
    def get_info(self):
        return {
            'model_type': 'SES',
            'smoothing_level': self.smoothing_level,
        }


class DendriticNeuronModel:
    """
    Dendritic Neuron Model (DNM) for time series forecasting
    Uses a neural network architecture inspired by dendritic computation
    """
    
    def __init__(self, input_dim=10, neurons=32, epochs=50, batch_size=16, verbose=True):
        """
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of lags)
        neurons : int
            Number of neurons in hidden layer
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler = MinMaxScaler()
        self.data_scaler = MinMaxScaler()
    
    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """Fit DNM model"""
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        
        if self.verbose:
            print(f"Fitting Dendritic Neuron Model...")
        
        # Scale data
        data_scaled = self.data_scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(data_scaled, self.input_dim)
        
        # Build neural network
        self.model = keras.Sequential([
            keras.layers.Dense(self.neurons, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train model
        self.model.fit(X, y, 
                      epochs=self.epochs, 
                      batch_size=self.batch_size,
                      verbose=0 if not self.verbose else 1)
        
        self.last_sequence = data_scaled[-self.input_dim:]
        
        return self
    
    def predict(self, steps=1):
        """Make predictions"""
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X_test = current_sequence.reshape(1, -1)
            pred_scaled = self.model.predict(X_test, verbose=0)[0, 0]
            
            # Inverse scale
            pred = self.data_scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        return np.array(predictions)
    
    def get_info(self):
        return {
            'model_type': 'DNM',
            'neurons': self.neurons,
            'epochs': self.epochs,
            'input_dim': self.input_dim,
        }
