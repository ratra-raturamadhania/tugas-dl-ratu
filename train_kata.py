# File: train_kata.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle # Pustaka untuk menyimpan objek tokenizer

MODEL_FILENAME = 'model_bi_lstm_kata.h5'
TOKENIZER_FILENAME = 'tokenizer_kata.pkl'

def train_and_save_bi_lstm_model():
    print("Memulai proses pelatihan model Prediksi Kata (Bi-LSTM)...")

    # =========================================================
    # 1. DATASET TEKS (Corpus Dummy)
    # =========================================================
    # Ganti dengan dataset teks yang lebih besar dan relevan jika ada.
    corpus = [
        "Saya suka belajar deep learning",
        "Deep learning adalah bidang yang menarik",
        "Belajar keras untuk tugas kuliah",
        "Kuliah ini sangat menantang dan seru"
    ]

    # =========================================================
    # 2. TOKENIZER DAN PREPROCESSING
    # =========================================================
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    # Simpan Tokenizer
    with open(TOKENIZER_FILENAME, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Tokenizer berhasil disimpan sebagai: {TOKENIZER_FILENAME}")

    # Buat Sequences: cth: ["saya suka", "saya suka belajar", ...]
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    
    # Padding: Membuat semua sequence panjangnya sama
    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X = padded_sequences[:, :-1] # Input: Semua kecuali kata terakhir
    labels = padded_sequences[:, -1] # Target: Kata terakhir
    y = to_categorical(labels, num_classes=total_words) # One-hot encoding target

    # =========================================================
    # 3. BANGUN MODEL BI-LSTM
    # =========================================================
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        Bidirectional(LSTM(150)), # Bidirectional LSTM
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # =========================================================
    # 4. LATIH DAN SIMPAN
    # =========================================================
    print(f"Melatih model dengan {len(X)} sampel. Epochs=50...")
    model.fit(X, y, epochs=50, verbose=0) 
    
    model.save(MODEL_FILENAME)
    print(f"\n✅ Model Prediksi Kata berhasil dilatih dan disimpan sebagai: {MODEL_FILENAME}")
    
if __name__ == '__main__':
    train_and_save_bi_lstm_model()