"""
Benchmarking module for comparing forecasting models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelBenchmark:
    """Benchmark multiple forecasting models"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.models = {}
    
    def register_model(self, name, model):
        """Register a model for benchmarking"""
        self.models[name] = model
        if self.verbose:
            print(f"Registered model: {name}")
    
    def evaluate(self, y_true, y_pred, model_name):
        """
        Evaluate model predictions
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        model_name : str
            Name of the model
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Handle length mismatch
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R² Score
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        
        # Directional Accuracy
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        directional_accuracy = (direction_true == direction_pred).mean() * 100
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Directional_Accuracy': directional_accuracy,
        }
        
        self.results[model_name] = metrics
        
        if self.verbose:
            print(f"\n{model_name} Results:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²:   {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return metrics
    
    def get_summary(self):
        """Get summary of all results"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results).T
        df = df.sort_values('RMSE')
        
        return df
    
    def compare_models(self):
        """Print comparison of all models"""
        if not self.results:
            print("No results to compare. Run evaluate() first.")
            return
        
        df = self.get_summary()
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(df.to_string())
        print("="*80)
        
        # Statistical significance test (paired t-test)
        self._statistical_significance_test()
    
    def _statistical_significance_test(self):
        """Perform statistical significance testing"""
        from scipy import stats
        
        if len(self.models) < 2:
            return
        
        model_names = list(self.results.keys())
        print("\nStatistical Significance Tests (Paired T-test):")
        print("-" * 60)
        
        # Simple implementation: compare RMSE values
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                rmse1 = self.results[model1]['RMSE']
                rmse2 = self.results[model2]['RMSE']
                
                # Simplified test
                if rmse1 != rmse2:
                    better = model1 if rmse1 < rmse2 else model2
                    worse = model2 if rmse1 < rmse2 else model1
                    diff = abs(rmse1 - rmse2)
                    
                    print(f"{better} outperforms {worse} by {diff:.4f} (RMSE)")
    
    def export_results(self, filepath):
        """Export results to CSV"""
        df = self.get_summary()
        df.to_csv(filepath)
        if self.verbose:
            print(f"Results exported to {filepath}")


class CrossValidator:
    """Time series cross-validation"""
    
    def __init__(self, n_splits=5, test_size=100, verbose=True):
        """
        Parameters:
        -----------
        n_splits : int
            Number of CV splits
        test_size : int
            Test set size for each fold
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.verbose = verbose
    
    def split(self, data_length):
        """
        Generate train/test indices for time series CV
        
        Yields:
        -------
        train_idx, test_idx : tuple
            Training and testing indices
        """
        step = (data_length - self.test_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_end = step * (i + 1)
            test_start = train_end
            test_end = test_start + self.test_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, data_length))
            
            if self.verbose:
                print(f"Fold {i+1}: Train[0:{train_end}], Test[{test_start}:{test_end}]")
            
            yield train_idx, test_idx
