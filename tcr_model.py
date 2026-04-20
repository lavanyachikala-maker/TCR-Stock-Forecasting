"""
T-Cell Receptor (TCR) Algorithm for Multivariate Time Series Forecasting
Inspired by the immune system's selective and adaptive behaviour
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TCRForecaster:
    """
    T-Cell Receptor Algorithm for Multivariate Time Series Forecasting
    
    The TCR algorithm uses Orthogonal Least Squares (OLS) for dynamic regressor
    selection and Inter-Cell Cohesion Factors (ICCF) to quantify each regressor's
    contribution to error reduction.
    """
    
    def __init__(self, max_lags=10, threshold=0.001, verbose=True):
        """
        Initialize TCR Forecaster
        
        Parameters:
        -----------
        max_lags : int
            Maximum number of lags to consider for feature engineering
        threshold : float
            Minimum error reduction (ICCF) to include a regressor
        verbose : bool
            Print progress information
        """
        self.max_lags = max_lags
        self.threshold = threshold
        self.verbose = verbose
        self.selected_regressors = []
        self.iccf_scores = []
        self.scaler = StandardScaler()
        self.model = None
        self.training_loss = None
        
    def create_lagged_features(self, data, max_lags):
        """
        Create lagged features for multivariate time series
        
        Parameters:
        -----------
        data : np.ndarray
            Input data of shape (n_samples, n_variables)
        max_lags : int
            Number of lags to create
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix of shape (n_samples - max_lags, n_features)
        y : np.ndarray
            Target vector of shape (n_samples - max_lags,)
        """
        n_vars = data.shape[1]
        X = []
        y = []
        
        for t in range(max_lags, len(data)):
            # Create feature vector from all lags of all variables
            lag_features = []
            for lag in range(1, max_lags + 1):
                lag_features.extend(data[t - lag, :])
            X.append(lag_features)
            y.append(data[t, 0])  # Predict first variable (closing price)
        
        return np.array(X), np.array(y)
    
    def orthogonal_least_squares_selection(self, X, y):
        """
        Orthogonal Least Squares (OLS) for dynamic regressor selection
        
        Selects regressors that maximize error reduction using
        Inter-Cell Cohesion Factors (ICCF)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        selected_indices : list
            Indices of selected regressors
        """
        selected_indices = []
        remaining_indices = list(range(X.shape[1]))
        current_error = np.sum(y ** 2)
        
        if self.verbose:
            print(f"Starting OLS selection. Initial error: {current_error:.4f}")
        
        iteration = 0
        while remaining_indices and len(selected_indices) < self.max_lags:
            best_idx = None
            best_error_reduction = 0
            best_iccf = 0
            
            # Try each remaining regressor
            for idx in remaining_indices:
                # Fit model with current selected + this candidate
                X_selected = X[:, selected_indices + [idx]]
                try:
                    model = LinearRegression()
                    model.fit(X_selected, y)
                    predictions = model.predict(X_selected)
                    error = np.sum((y - predictions) ** 2)
                    error_reduction = current_error - error
                    
                    # Inter-Cell Cohesion Factor (ICCF)
                    # Measures contribution ratio of this regressor
                    iccf = error_reduction / (current_error + 1e-10)
                    
                    if iccf > best_iccf and iccf > self.threshold:
                        best_error_reduction = error_reduction
                        best_iccf = iccf
                        best_idx = idx
                except:
                    continue
            
            if best_idx is None:
                if self.verbose:
                    print(f"OLS stopped at iteration {iteration}. No more regressors meet threshold.")
                break
            
            # Add best regressor
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            current_error -= best_error_reduction
            self.iccf_scores.append(best_iccf)
            
            iteration += 1
            if self.verbose:
                print(f"Iteration {iteration}: Selected regressor {best_idx}, ICCF: {best_iccf:.4f}, Remaining Error: {current_error:.4f}")
        
        if self.verbose:
            print(f"OLS selection complete. Selected {len(selected_indices)} regressors.")
        
        return selected_indices
    
    def fit(self, data):
        """
        Train the TCR model
        
        Parameters:
        -----------
        data : np.ndarray or pd.DataFrame
            Training data of shape (n_samples, n_variables)
            
        Returns:
        --------
        self : TCRForecaster
            Fitted model
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if self.verbose:
            print(f"Training TCR with data shape: {data.shape}")
        
        # Normalize data
        data_scaled = self.scaler.fit_transform(data)
        
        # Create lagged features
        X, y = self.create_lagged_features(data_scaled, self.max_lags)
        
        if self.verbose:
            print(f"Created lagged features: X shape {X.shape}, y shape {y.shape}")
        
        # Perform OLS-based regressor selection
        self.selected_regressors = self.orthogonal_least_squares_selection(X, y)
        
        # Train final model with selected regressors
        if len(self.selected_regressors) == 0:
            self.selected_regressors = list(range(min(5, X.shape[1])))
            if self.verbose:
                print(f"Using default regressors: {self.selected_regressors}")
        
        X_selected = X[:, self.selected_regressors]
        self.model = LinearRegression()
        self.model.fit(X_selected, y)
        
        # Store training loss
        self.training_loss = mean_squared_error(y, self.model.predict(X_selected))
        
        if self.verbose:
            print(f"Training complete. Training RMSE: {np.sqrt(self.training_loss):.4f}")
        
        return self
    
    def predict(self, data, steps=1):
        """
        Forecast future values
        
        Parameters:
        -----------
        data : np.ndarray or pd.DataFrame
            Input data for prediction
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        predictions : np.ndarray
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        predictions = []
        data_scaled = self.scaler.transform(data)
        
        # Use the last max_lags observations as starting point
        current_sequence = data_scaled[-self.max_lags:, :].copy()
        
        for step in range(steps):
            # Extract features from current sequence
            lag_features = []
            for lag in range(1, self.max_lags + 1):
                lag_features.extend(current_sequence[-lag, :])
            
            # Predict next value using selected regressors
            lag_features_array = np.array(lag_features)
            X_test = lag_features_array[self.selected_regressors].reshape(1, -1)
            pred_scaled = self.model.predict(X_test)[0]
            
            # Inverse transform to original scale
            pred_full = np.zeros((1, data.shape[1]))
            pred_full[0, 0] = pred_scaled
            pred_original = self.scaler.inverse_transform(pred_full)[0, 0]
            
            predictions.append(pred_original)
            
            # Update sequence with new prediction (scaled)
            new_row = np.zeros((1, data.shape[1]))
            new_row[0, 0] = pred_scaled
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def get_info(self):
        """Get model information"""
        info = {
            'model_type': 'TCR (T-Cell Receptor)',
            'selected_regressors': self.selected_regressors,
            'n_selected': len(self.selected_regressors),
            'iccf_scores': self.iccf_scores,
            'max_lags': self.max_lags,
            'threshold': self.threshold,
            'training_loss': self.training_loss,
        }
        return info
