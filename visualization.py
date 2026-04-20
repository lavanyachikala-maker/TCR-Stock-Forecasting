"""
Visualization module for plotting forecasts and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ForecastPlotter:
    """Plot forecasts and analysis"""
    
    def __init__(self, figsize=(15, 6), style='seaborn-v0_8-darkgrid'):
        self.figsize = figsize
        plt.style.use(style)
        self.figs = []
    
    def plot_forecast(self, actual, predictions_dict, title="Stock Price Forecast", 
                     save_path=None):
        """
        Plot actual vs predicted prices
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual prices
        predictions_dict : dict
            Dictionary of model_name: predictions
        title : str
            Plot title
        save_path : str
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual
        ax.plot(actual, 'k-', linewidth=2, label='Actual', zorder=3)
        
        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
        
        for (model_name, predictions), color in zip(predictions_dict.items(), colors):
            ax.plot(predictions, '--', linewidth=2, label=model_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved forecast plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, ax
    
    def plot_residuals(self, actual, predictions_dict, save_path=None):
        """Plot residuals for each model"""
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(self.figsize[0], 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for (model_name, predictions), ax in zip(predictions_dict.items(), axes):
            residuals = actual[:len(predictions)] - predictions
            
            ax.plot(residuals, 'b-', linewidth=1)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.set_ylabel('Residuals ($)', fontsize=10)
            ax.set_title(f'{model_name} - Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved residuals plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, axes
    
    def plot_error_distribution(self, actual, predictions_dict, save_path=None):
        """Plot error distribution boxplots"""
        errors = {}
        
        for model_name, predictions in predictions_dict.items():
            error = np.abs(actual[:len(predictions)] - predictions)
            errors[model_name] = error
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create boxplot
        data_for_box = [errors[name] for name in errors.keys()]
        bp = ax.boxplot(data_for_box, labels=errors.keys(), patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(errors)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Absolute Error ($)', fontsize=12)
        ax.set_title('Error Distribution by Model', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved error distribution plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, ax
    
    def plot_autocorrelation(self, data, lags=40, save_path=None):
        """Plot autocorrelation function (ACF)"""
        from statsmodels.graphics.tsaplots import plot_acf
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        plot_acf(data, lags=lags, ax=ax)
        ax.set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('ACF', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ACF plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, ax
    
    def plot_volatility(self, prices, window=20, save_path=None):
        """Plot rolling volatility"""
        log_returns = np.log(prices / np.roll(prices, 1))[1:]
        volatility = pd.Series(log_returns).rolling(window=window).std().values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Price
        ax1.plot(prices, 'b-', linewidth=2)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('Stock Price and Volatility', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Volatility
        ax2.plot(volatility, 'r-', linewidth=2)
        ax2.set_ylabel('Volatility', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved volatility plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, (ax1, ax2)
    
    def plot_metrics_comparison(self, results_df, metric='RMSE', save_path=None):
        """Plot metrics comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        results_df = results_df.sort_values(metric)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(results_df)))
        
        bars = ax.barh(range(len(results_df)), results_df[metric], color=colors)
        ax.set_yticks(range(len(results_df)))
        ax.set_yticklabels(results_df.index, fontsize=11)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                   f'{bar.get_width():.4f}', ha='left', va='center', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, ax
    
    def plot_returns_distribution(self, prices, bins=50, save_path=None):
        """Plot returns distribution"""
        returns = np.diff(prices) / prices[:-1]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(returns, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(returns), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        ax.axvline(np.median(returns), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
        
        ax.set_xlabel('Returns', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved returns distribution plot to {save_path}")
        
        self.figs.append(fig)
        plt.show()
        
        return fig, ax
    
    def close_all(self):
        """Close all figures"""
        for fig in self.figs:
            plt.close(fig)
        self.figs = []
