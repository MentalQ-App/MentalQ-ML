from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import logging

# Inisialisasi Flask
app = Flask(__name__)

# Aktifkan logging untuk debugging
logging.basicConfig(level=logging.DEBUG)

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load preprocessed data untuk tokenizer dan label encoder
df_data = pd.read_csv('data_hasil_prepos.csv')

# Handle NaN dan pastikan semua nilai di 'statement prepos' adalah string
df_data['statement prepos'] = df_data['statement prepos'].fillna('').astype(str)

# Inisialisasi dan fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_data['statement prepos'])

# Label urut sesuai output model
labels = ["Normal","Depression", "Suicidal", "Anxiety", "Stress", "Bipolar", "Personality Disorder"]

# Inisialisasi label encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Parameter preprocessing
vocab_size = 74564
max_length = 138
padding = 'post'
trunc_type = 'post'

# Fungsi untuk preprocessing input text
def preprocess_input_text(statements):
    try:
        # Validasi bahwa semua input adalah string
        statements = [str(statement) for statement in statements if statement]
        logging.debug(f"Valid statements: {statements}")
        
        # Tokenisasi
        sequences = tokenizer.texts_to_sequences(statements)
        logging.debug(f"Tokenized sequences: {sequences}")
        
        # Padding
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding, truncating=trunc_type)
        logging.debug(f"Padded sequences: {padded}")
        
        # Konversi ke float32
        return np.array(padded, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise e

# Endpoint API untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log permintaan yang masuk
        logging.debug(f"Request JSON: {request.json}")
        
        # Ambil data dari request
        input_data = request.json
        if 'data' not in input_data:
            return jsonify({'error': 'Input data should contain a key "data".'}), 400

        # Ekstrak data
        statements = input_data['data']
        if not isinstance(statements, list):
            return jsonify({'error': 'Data should be a list of strings.'}), 400

        # Preprocessing
        preprocessed_data = preprocess_input_text(statements)
        logging.debug(f"Preprocessed input to model: {preprocessed_data}")

        # Prediksi menggunakan model
        predictions = model.predict(preprocessed_data)
        logging.debug(f"Raw predictions: {predictions}")

        # Dekode prediksi menjadi label
        response = []
        threshold = 0.5  # Atur threshold untuk mendeteksi suicidal
        for i, statement in enumerate(statements):
            confidence_mapping = {labels[j]: float(predictions[i][j]) for j in range(len(labels))}
            predicted_status = labels[np.argmax(predictions[i])]

            # Logika untuk menangkap kasus suicidal
            if confidence_mapping["Suicidal"] >= threshold:
                predicted_status = "Suicidal"
            
            # Tambahkan heuristik berbasis keyword
            suicidal_keywords = ["kill", "end life", "suicide", "depressed", "hopeless"]
            if any(keyword in statement.lower() for keyword in suicidal_keywords):
                predicted_status = "Suicidal"

            response.append({
                'statement': statement,
                'predicted_status': predicted_status,
                'confidence_scores': confidence_mapping
            })

        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Jalankan Flask app
if __name__ == '__main__':
    app.run(debug=True)
