"""
Data integration module for fetching and preprocessing stock data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class StockDataFetcher:
    """Fetch and preprocess stock market data"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data = None
        self.ticker = None
    
    def fetch(self, ticker, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., '^GSPC', 'INFY')
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        data : pd.DataFrame
            Stock price data with columns: Open, High, Low, Close, Volume
        """
        if self.verbose:
            print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            self.data = data
            self.ticker = ticker
            
            if self.verbose:
                print(f"Successfully fetched {len(data)} trading days for {ticker}")
                print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_closing_prices(self):
        """Get closing prices as numpy array"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch() first.")
        
        return self.data['Close'].values
    
    def get_ohlcv(self):
        """Get OHLCV (Open, High, Low, Close, Volume)"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch() first.")
        
        return self.data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    def train_test_split(self, test_size=0.2, use_column='Close'):
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        use_column : str
            Column to use ('Close' by default)
            
        Returns:
        --------
        train_data : np.ndarray
            Training data
        test_data : np.ndarray
            Test data
        split_index : int
            Index where split occurs
        """
        data = self.data[use_column].values
        split_index = int(len(data) * (1 - test_size))
        
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        if self.verbose:
            print(f"Train/Test split: {len(train_data)} train, {len(test_data)} test")
            print(f"Train period: {self.data.index[0].date()} to {self.data.index[split_index-1].date()}")
            print(f"Test period: {self.data.index[split_index].date()} to {self.data.index[-1].date()}")
        
        return train_data, test_data, split_index
    
    def calculate_volatility(self, window=20):
        """Calculate rolling volatility"""
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        volatility = log_returns.rolling(window=window).std()
        return volatility.values
    
    def get_statistics(self):
        """Get data statistics"""
        prices = self.data['Close'].values
        returns = np.diff(prices) / prices[:-1]
        
        stats = {
            'ticker': self.ticker,
            'n_samples': len(prices),
            'price_min': prices.min(),
            'price_max': prices.max(),
            'price_mean': prices.mean(),
            'price_std': prices.std(),
            'returns_mean': returns.mean(),
            'returns_std': returns.std(),
            'returns_skewness': pd.Series(returns).skew(),
            'returns_kurtosis': pd.Series(returns).kurtosis(),
        }
        
        return stats


class DataPreprocessor:
    """Preprocess data for modeling"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, data):
        """Fit and transform data"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.scaler.fit(data)
        self.is_fitted = True
        
        return self.scaler.transform(data)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        return self.scaler.inverse_transform(data)
    
    def remove_outliers(self, data, z_threshold=3):
        """Remove outliers using z-score"""
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(data))
        return data[z_scores < z_threshold]
    
    def detrend(self, data):
        """Remove trend from data"""
        from scipy import signal
        
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        
        return signal.detrend(data)
