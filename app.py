import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# --- LOAD MODELS (Hanya TensorFlow) ---
model_logika = None
model_kata = None
model_saham = None
tokenizer_kata = None

def load_all_models():
    global model_logika, model_kata, model_saham, tokenizer_kata
    try:
        # 1. Load Model XOR
        model_logika = load_model('model_logika_xor.h5')
        
        # 2. Load Model Kata & Tokenizer
        model_kata = load_model('model_bi_lstm_kata.h5')
        with open('tokenizer_kata.pkl', 'rb') as handle:
            tokenizer_kata = pickle.load(handle)
            
        # 3. Load Model Saham
        model_saham = load_model('model_lstm_saham.h5')
        
        print("✅ Model Deep Learning Ratu Berhasil Dimuat (Tanpa Transformer)!")
    except Exception as e:
        print(f"❌ Error muat model: {e}")

load_all_models()

@app.route('/predict_logika', methods=['POST'])
def predict_logika():
    data = request.get_json()
    inp = np.array([data['input']], dtype=np.float32)
    pred = model_logika.predict(inp, verbose=0)
    return jsonify({"hasil": int(pred[0][0] > 0.5)})

@app.route('/prediksi_kata', methods=['POST'])
def prediksi_kata():
    data = request.get_json()
    input_text = data.get('sequence', '')
    token_list = tokenizer_kata.texts_to_sequences([input_text])[0]
    input_padded = pad_sequences([token_list], maxlen=20, padding='pre')
    predicted = np.argmax(model_kata.predict(input_padded, verbose=0), axis=-1)[0]
    output_word = next((w for w, i in tokenizer_kata.word_index.items() if i == predicted), "")
    return jsonify({"kata_prediksi": output_word})

@app.route('/prediksi_saham', methods=['POST'])
def prediksi_saham():
    data = request.get_json()
    input_data = data.get('data_historis')
    input_array = np.array(input_data).reshape(1, 10, 1)
    prediksi = model_saham.predict(input_array, verbose=0)
    return jsonify({"hasil": round(float(prediksi[0][0]), 2)})

@app.route('/compare_llm', methods=['POST'])
def compare_llm():
    data = request.get_json()
    input_text = data.get('text', '')
    # Versi ringan: Hanya jalankan Bi-LSTM saja
    token_list = tokenizer_kata.texts_to_sequences([input_text])[0]
    input_padded = pad_sequences([token_list], maxlen=20, padding='pre')
    pred_idx = np.argmax(model_kata.predict(input_padded, verbose=0), axis=-1)[0]
    res_lstm = next((w for w, i in tokenizer_kata.word_index.items() if i == pred_idx), "??")
    
    return jsonify({
        "bi_lstm": res_lstm,
        "transformer": "Nonaktif (Melebihi Kapasitas Hosting)"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
