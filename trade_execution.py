import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TradeExecutor:
    def __init__(self, model, astro_collector):
        self.model = model
        self.astro_collector = astro_collector
        self.symbol = "XAUUSD"
        self.lot_size = 0.1
        
        # Connect to MT5
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            raise Exception("Could not initialize MT5")
            
        # Check if symbol exists
        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise Exception(f"Symbol {self.symbol} not found")
            
    def get_current_price(self):
        """Get current bid/ask price"""
        symbol_info = mt5.symbol_info_tick(self.symbol)
        bid = symbol_info.bid
        ask = symbol_info.ask
        return {'bid': bid, 'ask': ask}
    
    def calculate_tp_sl(self, order_type, current_price, astrological_strength):
        """Calculate Take Profit and Stop Loss based on astrological strength"""
        # Base pips for TP and SL
        base_tp_pips = 50
        base_sl_pips = 30
        
        # Adjust based on astrological strength (0-1)
        tp_pips = base_tp_pips * (1 + astrological_strength)
        sl_pips = base_sl_pips * (1 - astrological_strength * 0.3)  # Less impact on SL
        
        # Convert pips to price for gold
        pip_value = 0.01  # For gold, 1 pip = 0.01
        
        if order_type == "BUY":
            entry = current_price['ask']
            tp = entry + (tp_pips * pip_value)
            sl = entry - (sl_pips * pip_value)
        else:  # SELL
            entry = current_price['bid']
            tp = entry - (tp_pips * pip_value)
            sl = entry + (sl_pips * pip_value)
            
        return tp, sl
    
    def place_order(self, order_type, lot_size=None, tp=None, sl=None, astrological_strength=0.5):
        """Place an order in MT5"""
        if lot_size is None:
            lot_size = self.lot_size
            
        current_price = self.get_current_price()
        
        # If TP and SL are not provided, calculate them
        if tp is None or sl is None:
            tp, sl = self.calculate_tp_sl(order_type, current_price, astrological_strength)
            
        # Set up order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": current_price['ask'] if order_type == "BUY" else current_price['bid'],
            "sl": sl,
            "tp": tp,
            "deviation": 5,  # Maximum price slippage in points
            "magic": 234000,  # Magic number for identification
            "comment": "OmniPattern AstroTrader Pro",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        # Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}")
            return None
        else:
            print(f"Order {order_type} placed successfully, order_id={result.order}")
            return result.order
    
    def get_live_data_for_prediction(self):
        """Get the latest data for model prediction"""
        # Get latest candles
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)  # Get 2 days of data to ensure we have enough
        
        # Get price data
        price_data = self.astro_collector.get_xauusd_data(start_time, end_time)
        
        # Get astrological data for each candle
        full_data = []
        for idx, row in price_data.iterrows():
            candle_time = row['time']
            
            # Convert to skyfield time
            ts = load.timescale()
            t = ts.utc(candle_time.year, candle_time.month, candle_time.day, 
                    candle_time.hour, candle_time.minute, candle_time.second)
            
            # Get planetary positions
            features = {
                'time': candle_time,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['tick_volume'],
                # Get all planetary features
                # ... (same as in AstroDataCollector.create_features_dataset)
            }
            
            full_data.append(features)
        
        df = pd.DataFrame(full_data)
        
        # Preprocess data
        processed_data = self.model.preprocess_data(df)
        
        # Create sequences
        X_live, _, _ = self.model.create_sequences(processed_data, future_steps=1)
        
        return X_live[-1:], df['close'].values[-1]  # Return latest sequence and current close price
    
    def generate_signal(self):
        """Generate trading signal based on model prediction"""
        # Get data for prediction
        X_live, current_price = self.get_live_data_for_prediction()
        
        # Make prediction
        price_pred, signal_pred, reversal_prob = self.model.signal_model.predict(X_live)
        
        # Process signal (softmax output: [sell, hold, buy])
        signal_idx = np.argmax(signal_pred[0])
        signal_labels = ['SELL', 'HOLD', 'BUY']
        signal = signal_labels[signal_idx]
        confidence = signal_pred[0][signal_idx]
        
        # Calculate predicted move
        predicted_move = price_pred[0][0] - current_price
        reversal_probability = reversal_prob[0][0]
        
        return {
            'current_price': current_price,
            'predicted_price': price_pred[0][0],
            'predicted_move': predicted_move,
            'signal': signal,
            'confidence': confidence,
            'reversal_probability': reversal_probability
        }
    
    def run_live_trading(self, interval_seconds=60):
        """Run live trading with periodic signal generation"""
        print("Starting live trading session...")
        
        while True:
            try:
                # Generate signal
                signal_data = self.generate_signal()
                
                print(f"\nTime: {datetime.now()}")
                print(f"Current price: {signal_data['current_price']}")
                print(f"Predicted price: {signal_data['predicted_price']}")
                print(f"Signal: {signal_data['signal']} (Confidence: {signal_data['confidence']:.2f})")
                print(f"Reversal probability: {signal_data['reversal_probability']:.2f}")
                
                # Execute trade if confidence is high enough
                if signal_data['signal'] != 'HOLD' and signal_data['confidence'] > 0.7:
                    # Place order with astrological strength as reversal probability
                    self.place_order(
                        signal_data['signal'],
                        astrological_strength=signal_data['confidence']
                    )
                
                # Wait for next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in live trading: {str(e)}")
                time.sleep(interval_seconds)
    
    def close_all_positions(self):
        """Close all open positions"""
        positions = mt5.positions_get(symbol=self.symbol)
        
        if positions:
            for position in positions:
                # Create close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(self.symbol).bid if position.type == 0 else mt5.symbol_info_tick(self.symbol).ask,
                    "deviation": 5,
                    "magic": 234000,
                    "comment": "OmniPattern close position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                print(f"Position {position.ticket} closed, result code: {result.retcode}")