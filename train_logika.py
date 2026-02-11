# File: train_logika.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Nama file model yang akan dimuat oleh app.py
MODEL_FILENAME = 'model_logika_xor.h5' 

def train_and_save_xor_model():
    print("Memulai proses pelatihan model Logika XOR...")

    # Data Input (A, B) dan Output (A XOR B)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # Bangun Model ANN Sederhana
    model = Sequential([
        Dense(units=2, activation='relu', input_shape=(2,)), 
        Dense(units=1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Latih dan Simpan
    model.fit(X, y, epochs=500, verbose=0) 
    model.save(MODEL_FILENAME)
    print(f"\n✅ Model Logika XOR berhasil dilatih dan disimpan sebagai: {MODEL_FILENAME}")
    
    # Uji coba
    print("\nHasil Uji Coba:")
    for input_data in X:
        prediksi_float = model.predict(np.array([input_data]), verbose=0)[0][0]
        prediksi_bulat = round(prediksi_float) 
        print(f"Input {input_data}: Prediksi = {prediksi_bulat}")

if __name__ == '__main__':
    train_and_save_xor_model()
    