import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ==============================
# KONFIGURASI AWAL (STABIL)
# ==============================
st.set_page_config(page_title="Deteksi COVID-19", page_icon="🦠", layout="centered")
st.set_option('client.showErrorDetails', True)

# ==============================
# CACHE DATASET
# ==============================
@st.cache_data
def load_data():
    data = {
        'Demam': [1,1,1,0,1,0,1,0,1,0,0,1],
        'Batuk': [1,1,0,1,1,0,1,1,0,0,1,0],
        'Sesak_Nafas': [1,0,0,0,0,0,0,0,1,0,1,0],
        'Sakit_Tenggorokan': [0,1,1,1,0,1,0,0,0,0,0,0],
        'Kehilangan_Penciuman': [1,0,0,0,1,0,0,1,0,0,0,1],
        'Diagnosa': [
            'Positif','Positif','Negatif','Negatif',
            'Positif','Negatif','Positif','Positif',
            'Positif','Negatif','Positif','Positif'
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# ==============================
# CACHE MODEL
# ==============================
@st.cache_resource
def load_model(df):
    X = df[['Demam', 'Batuk', 'Sesak_Nafas', 'Sakit_Tenggorokan', 'Kehilangan_Penciuman']]
    y = df['Diagnosa']
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)
    return model

model = load_model(df)

# ==============================
# UI
# ==============================
st.title("🦠 Sistem Deteksi Gejala COVID-19")
st.markdown("---")
st.subheader("🩺 Masukkan Gejala Pasien")

demam = st.selectbox("Demam", ["Tidak", "Ya"], key="demam")
batuk = st.selectbox("Batuk", ["Tidak", "Ya"], key="batuk")
sesak = st.selectbox("Sesak Nafas", ["Tidak", "Ya"], key="sesak")
tenggorokan = st.selectbox("Sakit Tenggorokan", ["Tidak", "Ya"], key="tenggorokan")
penciuman = st.selectbox("Kehilangan Penciuman", ["Tidak", "Ya"], key="penciuman")

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Deteksi COVID-19"):
    input_data = pd.DataFrame([[ 
        1 if demam == "Ya" else 0,
        1 if batuk == "Ya" else 0,
        1 if sesak == "Ya" else 0,
        1 if tenggorokan == "Ya" else 0,
        1 if penciuman == "Ya" else 0
    ]], columns=['Demam','Batuk','Sesak_Nafas','Sakit_Tenggorokan','Kehilangan_Penciuman'])

    hasil = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("📋 Hasil Diagnosa")

    if hasil == "Positif":
        st.error("⚠️ POSITIF COVID-19")
        st.write("Disarankan segera melakukan pemeriksaan lanjutan.")
    else:
        st.success("✅ NEGATIF COVID-19")
        st.write("Tetap jaga kesehatan dan protokol medis.")
