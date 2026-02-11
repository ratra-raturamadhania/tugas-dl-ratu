# File: train_saham.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 

MODEL_FILENAME = 'model_lstm_saham.h5' 
LOOK_BACK = 10 

def create_and_save_lstm_model():
    print("Memulai proses pelatihan model Saham...")

    # --- Data Dummy untuk Contoh (100 data poin) ---
    np.random.seed(42)
    data_points = 100
    base_data = np.linspace(50, 150, data_points)
    data = (base_data + np.random.randn(data_points) * 5).reshape(-1, 1) 

    # Normalisasi Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    def create_dataset(dataset, look_back=LOOK_BACK):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0] 
            X.append(a)
            Y.append(dataset[i + look_back, 0]) 
        return np.array(X), np.array(Y)

    X, y = create_dataset(scaled_data, LOOK_BACK)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Bangun Model LSTM
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1) 
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Latih dan Simpan
    print(f"Melatih model dengan {len(X)} sampel...")
    model.fit(X, y, epochs=100, batch_size=32, verbose=1) 
    
    model.save(MODEL_FILENAME)
    print(f"\n✅ Model Saham berhasil dilatih dan disimpan sebagai: {MODEL_FILENAME}")
    
if __name__ == '__main__':
    create_and_save_lstm_model()