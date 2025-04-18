import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

class AstroTraderModel:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.categorical_encoder = None
        self.model = None
        self.seq_length = 64  # As per your specification
        
    def preprocess_data(self, data):
        """Preprocess and normalize data"""
        # Separate numerical and categorical features
        numerical_features = data.select_dtypes(include=['float64', 'int64']).copy()
        categorical_features = data.select_dtypes(include=['object']).copy()
        
        # Scale price data
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        numerical_features[price_cols] = self.price_scaler.fit_transform(numerical_features[price_cols])
        
        # Scale other numerical features
        non_price_cols = [col for col in numerical_features.columns if col not in price_cols + ['time']]
        numerical_features[non_price_cols] = self.feature_scaler.fit_transform(numerical_features[non_price_cols])
        
        # One-hot encode categorical features
        if not categorical_features.empty:
            self.categorical_encoder = OneHotEncoder(sparse=False)
            encoded_cats = self.categorical_encoder.fit_transform(categorical_features)
            encoded_cats_df = pd.DataFrame(
                encoded_cats, 
                columns=self.categorical_encoder.get_feature_names_out(categorical_features.columns),
                index=categorical_features.index
            )
            
            # Combine numerical and encoded categorical data
            processed_data = pd.concat([numerical_features, encoded_cats_df], axis=1)
        else:
            processed_data = numerical_features
            
        return processed_data
    
    def create_sequences(self, data, target_col='close', future_steps=1):
        """Create sequences for LSTM model"""
        sequences = []
        targets = []
        
        # Drop time column for modeling
        if 'time' in data.columns:
            times = data['time'].values
            data = data.drop('time', axis=1)
        else:
            times = None
            
        data_array = data.values
        
        # Find index of target column
        target_idx = data.columns.get_loc(target_col)
        
        for i in range(len(data) - self.seq_length - future_steps + 1):
            seq = data_array[i:i+self.seq_length]
            target = data_array[i+self.seq_length+future_steps-1, target_idx]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets), times
    
    def build_model(self, input_shape):
        """Build the LSTM model architecture"""
        # Complex stacked LSTM model
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        
        # Second LSTM layer
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.3))
        
        # Third LSTM layer
        model.add(LSTM(32))
        model.add(Dropout(0.3))
        
        # Dense layers for prediction
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def build_signal_model(self, base_model, input_shape):
        """Build a model that outputs buy/sell signals in addition to price predictions"""
        # Reuse the trained base model's layers
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = LSTM(32)(x)
        x = Dropout(0.3)(x)
        
        # Branch 1: Price prediction
        price_branch = Dense(16, activation='relu')(x)
        price_output = Dense(1, name='price_prediction')(price_branch)
        
        # Branch 2: Signal prediction (buy=1, hold=0, sell=-1)
        signal_branch = Dense(16, activation='relu')(x)
        signal_output = Dense(3, activation='softmax', name='signal')(signal_branch)
        
        # Branch 3: Trend reversal probability
        reversal_branch = Dense(16, activation='relu')(x)
        reversal_output = Dense(1, activation='sigmoid', name='reversal_prob')(reversal_branch)
        
        # Combine into single model
        signal_model = Model(inputs=inputs, outputs=[price_output, signal_output, reversal_output])
        
        # Compile with multiple loss functions
        signal_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'price_prediction': 'mse',
                'signal': 'categorical_crossentropy',
                'reversal_prob': 'binary_crossentropy'
            },
            metrics={
                'price_prediction': ['mae'],
                'signal': ['accuracy'],
                'reversal_prob': ['accuracy']
            }
        )
        
        self.signal_model = signal_model
        return signal_model
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model"""
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
            
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
            ]
        )
        
        return history
    
    def predict(self, data):
        """Make predictions with the model"""
        # Ensure data is preprocessed and in sequence format
        if not isinstance(data, np.ndarray) or len(data.shape) != 3:
            raise ValueError("Input must be preprocessed and in sequence format")
        
        return self.model.predict(data)
    
    def save(self, filepath):
        """Save the model"""
        self.model.save(filepath)
        
    def load(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model