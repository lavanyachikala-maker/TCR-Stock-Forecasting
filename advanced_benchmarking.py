"""
Advanced benchmarking for TCR-Informer and baseline models
Includes statistical significance testing and performance analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AdvancedBenchmark:
    """Advanced performance evaluation"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.predictions = {}
    
    def add_predictions(self, model_name, y_true, y_pred):
        """Add model predictions"""
        self.predictions[model_name] = (y_true, y_pred)
    
    def evaluate_all(self):
        """Evaluate all models"""
        for model_name, (y_true, y_pred) in self.predictions.items():
            self.results[model_name] = self._evaluate_model(y_true, y_pred, model_name)
    
    def _evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate single model"""
        # Handle length mismatch
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        dir_acc = (direction_true == direction_pred).mean() * 100
        
        # Volatility match
        vol_true = np.std(np.diff(y_true))
        vol_pred = np.std(np.diff(y_pred))
        vol_ratio = vol_pred / (vol_true + 1e-8)
        
        # Max drawdown match
        cum_ret_true = np.cumprod(1 + np.diff(y_true) / (y_true[:-1] + 1e-8))
        cum_ret_pred = np.cumprod(1 + np.diff(y_pred) / (y_pred[:-1] + 1e-8))
        
        dd_true = np.min(cum_ret_true / np.maximum.accumulate(cum_ret_true))
        dd_pred = np.min(cum_ret_pred / np.maximum.accumulate(cum_ret_pred))
        
        # Sharpe ratio (assuming 252 trading days/year)
        returns_true = np.diff(y_true) / (y_true[:-1] + 1e-8)
        returns_pred = np.diff(y_pred) / (y_pred[:-1] + 1e-8)
        
        sharpe_true = np.mean(returns_true) / (np.std(returns_true) + 1e-8) * np.sqrt(252)
        sharpe_pred = np.mean(returns_pred) / (np.std(returns_pred) + 1e-8) * np.sqrt(252)
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Directional_Acc': dir_acc,
            'Volatility_Ratio': vol_ratio,
            'Max_Drawdown_True': dd_true,
            'Max_Drawdown_Pred': dd_pred,
            'Sharpe_True': sharpe_true,
            'Sharpe_Pred': sharpe_pred,
        }
        
        if self.verbose:
            print(f"\n{model_name} Evaluation:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {dir_acc:.2f}%")
            print(f"  Volatility Ratio: {vol_ratio:.4f}")
            print(f"  Sharpe Ratio (True/Pred): {sharpe_true:.4f} / {sharpe_pred:.4f}")
        
        return metrics
    
    def get_summary(self):
        """Get summary dataframe"""
        return pd.DataFrame(self.results).T.sort_values('RMSE')
    
    def statistical_tests(self):
        """Perform statistical significance tests"""
        if len(self.predictions) < 2:
            print("Need at least 2 models for comparison")
            return
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        model_names = list(self.predictions.keys())
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                y_true1, y_pred1 = self.predictions[model1]
                y_true2, y_pred2 = self.predictions[model2]
                
                min_len = min(len(y_pred1), len(y_pred2))
                error1 = np.abs(y_true1[:min_len] - y_pred1[:min_len])
                error2 = np.abs(y_true2[:min_len] - y_pred2[:min_len])
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(error1, error2)
                
                print(f"\n{model1} vs {model2}:")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    better_model = model1 if np.mean(error1) < np.mean(error2) else model2
                    print(f"  ✓ {better_model} is significantly better (p < 0.05)")
                else:
                    print(f"  ✗ No significant difference (p >= 0.05)")
    
    def plot_comparison(self, metric='RMSE'):
        """Plot metric comparison"""
        df = self.get_summary()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(df)))
        bars = ax.barh(range(len(df)), df[metric], color=colors)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f'{bar.get_width():.4f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig, ax
    
    def export_results(self, filepath):
        """Export results"""
        self.get_summary().to_csv(filepath)
        print(f"Results saved to {filepath}")
