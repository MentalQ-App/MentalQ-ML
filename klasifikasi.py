import os
import warnings
import numpy as np
import h5py
import stanza
import re
import string
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Menonaktifkan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Menonaktifkan warnings dari stanza dan torch
warnings.filterwarnings("ignore", category=UserWarning, module="stanza")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# Model ML import
def load_trained_model():
    model = load_model('model_save_ml/ml_model_lstm.h5') 
    return model

# MOdel Embedding matrix import
def load_embedding():
    with h5py.File("model_word2vec/word2vec_model_MentalQ.h5", "r") as h5file:
        vocab = list(h5file['vocab'][:])
        vectors = np.array(h5file['vectors'][:])
    word_index = {word: i + 1 for i, word in enumerate(vocab)}
    return word_index

# Cleaning Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Stopword Removal
def remove_stopwords(text, stop_words):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# lemmatization
nlp = stanza.Pipeline('id', processors='tokenize,lemma', use_gpu=False)
def lemmatize_text(text):
    doc = nlp(text)
    sentences = []
    for sentence in doc.sentences:
        lemmas = [word.lemma for word in sentence.words]
        sentences.append(' '.join(lemmas))
    return ' '.join(sentences)

# Prepocessing
def preprocess_text(text, stop_words):
    
    text = clean_text(text)
    text = remove_stopwords(text, stop_words)
    text = lemmatize_text(text)
    tokens = word_tokenize(text)
    
    return tokens

# Text to Sequence
def text_to_sequence(tokens, word_index):
    return [word_index[word] for word in tokens if word in word_index]

# Prediksi text baru
def predict_new_text(model, word_index, stop_words, new_text):
    # Preprocessing
    tokens_new = preprocess_text(new_text, stop_words)
    sequence_new = text_to_sequence(tokens_new, word_index)
    
    # Padding sequence
    padded_sequence_new = pad_sequences([sequence_new], maxlen=100, padding='post')
    
    # Prediksi
    predictions = model.predict(padded_sequence_new)
    
    # Mengambil kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)
    
    # Menggunakan label encoder untuk mengubah hasil prediksi menjadi label asli
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('data/label_encoding.npy', allow_pickle=True)  # import one hot encoding label
    
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    return predicted_label[0], predictions[0][predicted_class[0]]

# Fungsi Utama
def main():
    # Model dan embedding
    model = load_trained_model()
    word_index = load_embedding()
    
    # Stopwords
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())
    
    # Input teks baru untuk klasifikasi
    new_text = input("Masukkan teks Jurnaling: ")
    
    # Prediksi
    predicted_label, probability = predict_new_text(model, word_index, stop_words, new_text)
    
    print(f"Teks: {new_text}")
    print(f"Prediksi Kelas: {predicted_label}")
    print(f"Probabilitas Kelas: {probability}")

if __name__ == "__main__":
    main()