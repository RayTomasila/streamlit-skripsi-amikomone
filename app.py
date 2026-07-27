import streamlit as st
import pickle
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download Stopwords NLTK (Hanya jika belum ada)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Testing Klasifikasi Teks Amikom One", layout="centered")

# STYLING (Force Light Mode: Background Putih, Teks Hitam)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #000000 !important;
    }
    textarea {
        color: #000000 !important;
        background-color: #F8F9FA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Klasifikasi Sentimen Ulasan Aplikasi - Amikom One")
st.write("Sistem inferensi *real-time* menggunakan **Support Vector Machine (Kernel Sigmoid)** - Model Terbaik Skenario 90:10.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SVM_PATH = os.path.join(BASE_DIR, "svm_best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
SLANG_PATH = os.path.join(BASE_DIR, "slang.csv") 

# Load File Pickle Model SVM & TF-IDF
@st.cache_resource
def load_saved_models():
    model_svm, vectorizer = None, None
    if os.path.exists(MODEL_SVM_PATH):
        with open(MODEL_SVM_PATH, "rb") as f: model_svm = pickle.load(f)
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    return model_svm, vectorizer

model_svm, vectorizer = load_saved_models()

# Load kamus slang
@st.cache_resource
def load_slang_dict():
    if os.path.exists(SLANG_PATH):
        slang_df = pd.read_csv(SLANG_PATH)
        return dict(zip(slang_df['slang'], slang_df['formal']))
    return None

slang_dict = load_slang_dict()

# KONFIGURASI PREPROCESSING
tokenizer = RegexpTokenizer(r'\w+')
stemmer = StemmerFactory().create_stemmer()
all_stopwords = set(stopwords.words('indonesian'))
kata_penting = {'tidak', 'bukan', 'kurang', 'belum', 'jangan', 'baik', 'tapi', 'banyak', 'sering', 'keluar', 'sendiri', 'masalah', 'error', 'eror', 'bisa', 'sulit', 'gagal', 'lama', 'lambat'}
all_stopwords = all_stopwords - kata_penting

def pipeline_preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = tokenizer.tokenize(text)
    if slang_dict:
        tokens = [slang_dict[token] if token in slang_dict else token for token in tokens]
        
    tokens = [t for t in tokens if t not in all_stopwords]
    
    new_tokens = []
    for token in tokens:
        if token == 'perbaiki':
            new_tokens.append('perbaiki')
        else:
            new_tokens.append(stemmer.stem(token))
            
    return " ".join(new_tokens)


if model_svm is None or vectorizer is None:
    st.error("File model (SVM) atau TF-IDF Vectorizer tidak ditemukan di folder!")
elif slang_dict is None:
    st.error("File 'slang.csv' tidak ditemukan di folder!")
else:
    user_input = st.text_area(
        "Masukkan teks ulasan untuk diuji:",
        placeholder="Contoh: Aplikasi sering keluar sendiri saat mau presensi, tolong diperbaiki dong dev..."
    )

    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Model sedang memproses preprocessing dan prediksi..."):
                
                # 1. Preprocessing
                text_clean = pipeline_preprocessing(user_input)
                st.info(f"**Hasil Preprocessing:** '{text_clean}'")
                
                # 2. Ekstraksi Fitur TF-IDF
                text_tfidf = vectorizer.transform([text_clean])
                
                # 3. Prediksi Model SVM
                pred_svm = model_svm.predict(text_tfidf)
                hasil_prediksi = pred_svm[0]
                
                st.markdown("### Hasil Prediksi Model:")
                
                # 4. Mapping 3 Kelas Sentimen
                # Asumsi output dari model lu berupa teks ('Positif', 'Netral', 'Negatif') 
                # atau angka (1, 0, -1). Logika di bawah mencakup keduanya.
                if hasil_prediksi == 'Positif' or hasil_prediksi == 1:
                    label_svm = "POSITIF"
                elif hasil_prediksi == 'Netral' or hasil_prediksi == 0:
                    label_svm =  "NETRAL"
                else:
                    label_svm = "NEGATIF"
                    
                st.metric(label="Kelas Sentimen (SVM Sigmoid 90:10):", value=label_svm)
                st.caption("Performa Keseluruhan Model | Akurasi: **83.56%**")