import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import custom modules
from data_collection import AstroDataCollector
from model_architecture import AstroTraderModel
from trade_execution import TradeExecutor
from dashboard import AstroTraderDashboard

def main():
    print("Initializing OmniPattern AstroTrader Pro...")
    
    # Create data collector
    astro_collector = AstroDataCollector()
    
    # Create or load model
    model_path = "models/astro_trader_model.h5"
    model = AstroTraderModel()
    
    # Check if model exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load(model_path)
    else:
        print("No existing model found. Training new model...")
        
        # Define training period
        train_start = datetime(2023, 3, 21, 22, 55, 0)
        train_end = datetime(2024, 10, 17, 14, 15, 0)
        
        print(f"Collecting data from {train_start} to {train_end}...")
        # Get training data
        data = astro_collector.create_features_dataset(train_start, train_end)
        
        # Preprocess data
        processed_data = model.preprocess_data(data)
        
        # Create sequences
        X, y, _ = model.create_sequences(processed_data)
        
        # Build and train model
        print(f"Building and training model with {len(X)} sequences...")
        model.build_model((X.shape[1], X.shape[2]))
        model.train(X, y, epochs=200, batch_size=64)
        
        # Build signal model
        model.build_signal_model(model.model, (X.shape[1], X.shape[2]))
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print("Model saved successfully.")
    
    # Create trade executor
    executor = TradeExecutor(model, astro_collector)
    
    # Initialize and run dashboard
    dashboard = AstroTraderDashboard(model, astro_collector, executor)
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()