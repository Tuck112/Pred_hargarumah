import streamlit as st
import pandas as pd
import joblib

# Memuat model yang telah dilatih
model_pipeline = joblib.load('heart_disease_model.pkl')

# Fungsi untuk menampilkan hasil prediksi dengan tampilan yang berbeda
def display_prediction_result(prediction):
    if prediction[0] == 1:
        st.error("Model memprediksi bahwa pasien kemungkinan memiliki penyakit jantung.")
    else:
        st.success("Model memprediksi bahwa pasien kemungkinan tidak memiliki penyakit jantung.")

# Aplikasi Streamlit
st.title("Prediksi Penyakit Jantung")

# Kolom input
age = st.slider("Usia", min_value=20, max_value=90, value=50)
sex = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
cp = st.selectbox("Tipe Nyeri Dada", ['Angina khas', 'Angina atipikal', 'Non-anginal', 'Asimtomatik'])
trestbps = st.slider("Tekanan Darah Istirahat (mm Hg)", min_value=90, max_value=200, value=120)
chol = st.slider("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", ['Ya', 'Tidak'])
restecg = st.selectbox("Hasil EKG Istirahat", ['Normal', 'Abnormalitas ST-T', 'Hipertrofi LV'])
thalch = st.slider("Denyut Jantung Maksimum yang Dicapai", min_value=60, max_value=220, value=150)
exang = st.selectbox("Angina yang Dipicu oleh Latihan", ['Ya', 'Tidak'])
oldpeak = st.slider("Depresi ST yang Dipicu oleh Latihan", min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox("Kemiringan Segmen ST Puncak Latihan", ['Naik', 'Datar', 'Menurun'])
ca = st.select_slider("Jumlah Pembuluh Besar yang Diberi Warna oleh Fluoroskopi", options=[0, 1, 2, 3])
thal = st.selectbox("Thalasemia", ['Normal', 'Defek Tetap', 'Defek Reversibel'])

# Melakukan prediksi saat tombol "Predict" diklik
if st.button("Prediksi"):
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Mengonversi dictionary input ke dalam dataframe
    input_df = pd.DataFrame([input_data])

    # Melakukan prediksi
    prediction = model_pipeline.predict(input_df)

    # Menampilkan hasil prediksi
    display_prediction_result(prediction)
