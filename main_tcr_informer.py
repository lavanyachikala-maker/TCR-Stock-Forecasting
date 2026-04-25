"""
Complete TCR-Informer Stock Forecasting Pipeline
Handles data fetching, model training, evaluation, and visualization
"""

import os
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from data_integration import StockDataFetcher, DataPreprocessor
from tcr_informer_model import TCRInformerForecaster
from baseline_models import ARIMAModel, SARIMAModel, SESModel, DendriticNeuronModel
from advanced_benchmarking import AdvancedBenchmark
from visualization import ForecastPlotter
import warnings
warnings.filterwarnings('ignore')


def main():
    """Complete TCR-Informer Pipeline"""
    
    print("="*80)
    print("TCR-INFORMER HYBRID STOCK FORECASTING SYSTEM")
    print("="*80)
    
    # Configuration
    TICKER = '^GSPC'  # S&P 500
    START_DATE = '2022-01-01'
    END_DATE = '2024-12-31'
    SEQ_LEN = 96
    LABEL_LEN = 48
    PRED_LEN = 24
    BATCH_SIZE = 32
    EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Sequence Length: {SEQ_LEN}")
    print(f"  Prediction Length: {PRED_LEN}")
    print(f"  Device: {DEVICE}")
    
    # ==================== FETCH DATA ====================
    print("\n[1] FETCHING DATA...")
    print("-" * 80)
    
    fetcher = StockDataFetcher(verbose=True)
    data = fetcher.fetch(TICKER, START_DATE, END_DATE)
    prices = fetcher.get_closing_prices()
    
    stats = fetcher.get_statistics()
    print(f"\nPrice Statistics:")
    print(f"  Min: ${stats['price_min']:.2f}")
    print(f"  Max: ${stats['price_max']:.2f}")
    print(f"  Mean: ${stats['price_mean']:.2f}")
    print(f"  Std: ${stats['price_std']:.2f}")
    
    # Prepare as 2D array [N, 1]
    prices_2d = prices.reshape(-1, 1)
    
    # ==================== BUILD AND TRAIN TCR-INFORMER ====================
    print("\n[2] BUILDING TCR-INFORMER MODEL...")
    print("-" * 80)
    
    forecaster = TCRInformerForecaster(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.1,
        learning_rate=0.0001,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        verbose=True
    )
    
    forecaster.build_model(enc_in=1, dec_in=1, c_out=1)
    
    print("\n[3] PREPARING DATA...")
    print("-" * 80)
    
    train_x, train_y, test_x, test_y = forecaster.prepare_data(prices_2d, train_ratio=0.8)
    
    print("\n[4] TRAINING TCR-INFORMER...")
    print("-" * 80)
    
    forecaster.train(train_x, train_y, val_x=test_x, val_y=test_y)
    
    print("\n[5] MAKING PREDICTIONS...")
    print("-" * 80)
    
    tcr_informer_pred, tcr_informer_true = forecaster.predict(test_x, test_y)
    tcr_informer_pred = tcr_informer_pred.flatten()
    tcr_informer_true = tcr_informer_true.flatten()
    
    print(f"TCR-Informer predictions shape: {tcr_informer_pred.shape}")
    print(f"True values shape: {tcr_informer_true.shape}")
    
    # ==================== BASELINE MODELS ====================
    print("\n[6] TRAINING BASELINE MODELS...")
    print("-" * 80)
    
    train_prices = prices[:int(len(prices) * 0.8)]
    test_prices = prices[int(len(prices) * 0.8):]
    
    predictions = {'TCR-Informer': (tcr_informer_true, tcr_informer_pred)}
    
    # ARIMA
    print("\nTraining ARIMA...")
    try:
        arima = ARIMAModel(order=(5, 1, 2), verbose=False)
        arima.fit(train_prices)
        arima_pred = arima.predict(steps=len(test_prices))
        predictions['ARIMA'] = (test_prices, arima_pred)
    except Exception as e:
        print(f"ARIMA training failed: {e}")
    
    # SARIMA
    print("Training SARIMA...")
    try:
        sarima = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), verbose=False)
        sarima.fit(train_prices)
        sarima_pred = sarima.predict(steps=len(test_prices))
        predictions['SARIMA'] = (test_prices, sarima_pred)
    except Exception as e:
        print(f"SARIMA training failed: {e}")
    
    # SES
    print("Training SES...")
    try:
        ses = SESModel(smoothing_level=0.2, verbose=False)
        ses.fit(train_prices)
        ses_pred = ses.predict(steps=len(test_prices))
        predictions['SES'] = (test_prices, ses_pred)
    except Exception as e:
        print(f"SES training failed: {e}")
    
    # DNM
    print("Training DNM...")
    try:
        dnm = DendriticNeuronModel(input_dim=SEQ_LEN, neurons=32, epochs=50, verbose=False)
        dnm.fit(train_prices.reshape(-1, 1))
        dnm_pred = dnm.predict(steps=len(test_prices))
        predictions['DNM'] = (test_prices, dnm_pred)
    except Exception as e:
        print(f"DNM training failed: {e}")
    
    # ==================== EVALUATION ====================
    print("\n[7] EVALUATING ALL MODELS...")
    print("-" * 80)
    
    benchmark = AdvancedBenchmark(verbose=True)
    
    for model_name, (y_true, y_pred) in predictions.items():
        benchmark.add_predictions(model_name, y_true, y_pred)
    
    benchmark.evaluate_all()
    benchmark.statistical_tests()
    
    summary_df = benchmark.get_summary()
    print("\n" + summary_df.to_string())
    
    # ==================== VISUALIZATION ====================
    print("\n[8] GENERATING VISUALIZATIONS...")
    print("-" * 80)
    
    os.makedirs('results', exist_ok=True)
    
    plotter = ForecastPlotter(figsize=(16, 8))
    
    # Forecast comparison
    print("Plotting forecasts...")
    pred_dict = {name: pred for name, (_, pred) in predictions.items()}
    plotter.plot_forecast(
        tcr_informer_true, pred_dict,
        title=f"{TICKER} - Stock Price Forecast Comparison",
        save_path='results/01_forecast_comparison.png'
    )
    
    # Metrics comparison
    print("Plotting metrics...")
    fig, ax = benchmark.plot_comparison('RMSE')
    fig.savefig('results/02_rmse_comparison.png', dpi=300, bbox_inches='tight')
    
    fig, ax = benchmark.plot_comparison('MAE')
    fig.savefig('results/03_mae_comparison.png', dpi=300, bbox_inches='tight')
    
    # Volatility analysis
    plotter.plot_volatility(prices, window=20, save_path='results/04_volatility.png')
    
    # Error distribution
    plotter.plot_error_distribution(tcr_informer_true, pred_dict, save_path='results/05_error_distribution.png')
    
    # ==================== EXPORT RESULTS ====================
    print("\n[9] EXPORTING RESULTS...")
    print("-" * 80)
    
    benchmark.export_results('results/model_comparison.csv')
    forecaster.save('results/tcr_informer_model.pt')
    
    # Summary report
    with open('results/summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TCR-INFORMER HYBRID MODEL - FORECASTING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Ticker: {TICKER}\n")
        f.write(f"  Period: {START_DATE} to {END_DATE}\n")
        f.write(f"  Sequence Length: {SEQ_LEN}\n")
        f.write(f"  Prediction Length: {PRED_LEN}\n")
        f.write(f"  Epochs: {EPOCHS}\n\n")
        
        f.write("RESULTS:\n")
        f.write(summary_df.to_string())
        f.write("\n\n")
        
        best_model = summary_df.index[0]
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Best RMSE: {summary_df.loc[best_model, 'RMSE']:.4f}\n")
        f.write(f"Best MAE: {summary_df.loc[best_model, 'MAE']:.4f}\n")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_model = summary_df.index[0]
    print(f"\n✓ Best Model: {best_model}")
    print(f"✓ Best RMSE: {summary_df.loc[best_model, 'RMSE']:.4f}")
    print(f"✓ Best MAE: {summary_df.loc[best_model, 'MAE']:.4f}")
    print(f"\n✓ Results saved to: results/")
    print(f"✓ Model saved to: results/tcr_informer_model.pt")
    
    print("\n" + "="*80)
    print("FORECASTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
