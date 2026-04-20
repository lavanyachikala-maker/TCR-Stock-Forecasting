"""
Main script to run the complete TCR Stock Forecasting pipeline
"""

import numpy as np
import pandas as pd
from config import *
from data_integration import StockDataFetcher, DataPreprocessor
from tcr_model import TCRForecaster
from baseline_models import ARIMAModel, SARIMAModel, SESModel, DendriticNeuronModel
from benchmarking import ModelBenchmark
from visualization import ForecastPlotter
import warnings
warnings.filterwarnings('ignore')


def main():
    """Run complete forecasting pipeline"""
    
    print("="*80)
    print("TCR STOCK FORECASTING SYSTEM")
    print("="*80)
    
    # ==================== DATA FETCHING ====================
    print("\n[1] FETCHING DATA...")
    print("-" * 80)
    
    fetcher = StockDataFetcher(verbose=True)
    data = fetcher.fetch(
        ticker=DATA_CONFIG['tickers'][0],  # S&P 500
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date']
    )
    
    # Get statistics
    stats = fetcher.get_statistics()
    print(f"\nData Statistics:")
    for key, value in stats.items():
        if key != 'ticker':
            print(f"  {key}: {value:.4f}")
    
    # ==================== DATA PREPROCESSING ====================
    print("\n[2] PREPROCESSING DATA...")
    print("-" * 80)
    
    prices = fetcher.get_closing_prices()
    train_data, test_data, split_idx = fetcher.train_test_split(test_size=0.2)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # ==================== MODEL TRAINING ====================
    print("\n[3] TRAINING MODELS...")
    print("-" * 80)
    
    # TCR Model
    print("\nTraining TCR Model...")
    tcr = TCRForecaster(max_lags=TCR_CONFIG['max_lags'], 
                       threshold=TCR_CONFIG['threshold'],
                       verbose=True)
    tcr_train_data = train_data.reshape(-1, 1)
    tcr.fit(tcr_train_data)
    
    # ARIMA Model
    print("\nTraining ARIMA Model...")
    arima = ARIMAModel(order=ARIMA_CONFIG['order'], verbose=True)
    arima.fit(train_data)
    
    # SARIMA Model
    print("\nTraining SARIMA Model...")
    sarima = SARIMAModel(
        order=SARIMA_CONFIG['order'],
        seasonal_order=SARIMA_CONFIG['seasonal_order'],
        verbose=True
    )
    sarima.fit(train_data)
    
    # SES Model
    print("\nTraining SES Model...")
    ses = SESModel(smoothing_level=SES_CONFIG['smoothing_level'], verbose=True)
    ses.fit(train_data)
    
    # DNM Model
    print("\nTraining DNM Model...")
    dnm = DendriticNeuronModel(
        input_dim=TCR_CONFIG['max_lags'],
        neurons=DNM_CONFIG['neurons'],
        epochs=DNM_CONFIG['epochs'],
        verbose=False
    )
    dnm_train_data = train_data.reshape(-1, 1)
    dnm.fit(dnm_train_data)
    
    # ==================== MAKING PREDICTIONS ====================
    print("\n[4] MAKING PREDICTIONS...")
    print("-" * 80)
    
    predictions = {}
    
    # TCR Predictions
    print("TCR predictions...")
    tcr_pred = tcr.predict(tcr_train_data, steps=len(test_data))
    predictions['TCR'] = tcr_pred
    
    # ARIMA Predictions
    print("ARIMA predictions...")
    arima_pred = arima.predict(steps=len(test_data))
    predictions['ARIMA'] = arima_pred
    
    # SARIMA Predictions
    print("SARIMA predictions...")
    sarima_pred = sarima.predict(steps=len(test_data))
    predictions['SARIMA'] = sarima_pred
    
    # SES Predictions
    print("SES predictions...")
    ses_pred = ses.predict(steps=len(test_data))
    predictions['SES'] = ses_pred
    
    # DNM Predictions
    print("DNM predictions...")
    dnm_pred = dnm.predict(steps=len(test_data))
    predictions['DNM'] = dnm_pred
    
    # ==================== EVALUATION ====================
    print("\n[5] EVALUATING MODELS...")
    print("-" * 80)
    
    benchmark = ModelBenchmark(verbose=True)
    
    for model_name, preds in predictions.items():
        benchmark.evaluate(test_data, preds, model_name)
    
    # Print summary
    benchmark.compare_models()
    summary_df = benchmark.get_summary()
    
    # ==================== VISUALIZATION ====================
    print("\n[6] GENERATING VISUALIZATIONS...")
    print("-" * 80)
    
    plotter = ForecastPlotter(figsize=PLOT_CONFIG['figsize'], 
                             style=PLOT_CONFIG['style'])
    
    # Plot 1: Forecast Comparison
    print("Plotting forecast comparison...")
    plotter.plot_forecast(test_data, predictions, 
                         title=f"{DATA_CONFIG['tickers'][0]} - Stock Price Forecast",
                         save_path='results/01_forecast_comparison.png')
    
    # Plot 2: Residuals
    print("Plotting residuals...")
    plotter.plot_residuals(test_data, predictions,
                          save_path='results/02_residuals.png')
    
    # Plot 3: Error Distribution
    print("Plotting error distribution...")
    plotter.plot_error_distribution(test_data, predictions,
                                   save_path='results/03_error_distribution.png')
    
    # Plot 4: Volatility
    print("Plotting volatility...")
    plotter.plot_volatility(prices, window=20,
                           save_path='results/04_volatility.png')
    
    # Plot 5: Autocorrelation
    print("Plotting autocorrelation...")
    plotter.plot_autocorrelation(test_data, lags=40,
                                save_path='results/05_autocorrelation.png')
    
    # Plot 6: Returns Distribution
    print("Plotting returns distribution...")
    plotter.plot_returns_distribution(prices, bins=50,
                                     save_path='results/06_returns_distribution.png')
    
    # Plot 7: Metrics Comparison
    print("Plotting metrics comparison...")
    plotter.plot_metrics_comparison(summary_df, metric='RMSE',
                                   save_path='results/07_metrics_rmse.png')
    
    plotter.plot_metrics_comparison(summary_df, metric='MAE',
                                   save_path='results/08_metrics_mae.png')
    
    # ==================== EXPORT RESULTS ====================
    print("\n[7] EXPORTING RESULTS...")
    print("-" * 80)
    
    # Save results to CSV
    summary_df.to_csv('results/model_comparison.csv')
    print("Saved model comparison to results/model_comparison.csv")
    
    # Save TCR model info
    tcr_info = tcr.get_info()
    print(f"\nTCR Model Info:")
    for key, value in tcr_info.items():
        print(f"  {key}: {value}")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_model = summary_df.index[0]
    best_rmse = summary_df.loc[best_model, 'RMSE']
    
    print(f"\nBest Model: {best_model}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"\nComplete results saved to results/ directory")
    
    print("\n" + "="*80)
    print("FORECASTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
