# === 1. Import Library ===
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === 2. Load Model Klasifikasi dan Regresi ===
model_cls, fitur_model = joblib.load("model_klasifikasi_petir3.pkl")
model_reg, fitur_model_reg = joblib.load("model_regresi_petir3.pkl")

# === 3. Streamlit Interface ===
st.set_page_config(page_title="Prediksi Petir & Jumlah Sambaran CG", layout="centered")
st.title("🌩️ Prediksi Kejadian Petir dan Estimasi Jumlah Sambaran CG")
st.markdown("Masukkan parameter atmosfer berikut:")

# === 4. Form Input User ===
with st.form("form_input"):
    LI = st.number_input("Lifted Index (LI)", value=-2.0)
    SWEAT = st.number_input("SWEAT Index", value=200.0)
    KI = st.number_input("K Index", value=30.0)
    TTI = st.number_input("Total Totals Index (TTI)", value=48.0)
    CAPE = st.number_input("CAPE (J/kg)", value=1000.0)
    SI = st.number_input("Showalter Index (SI)", value=1.0)
    PW = st.number_input("Precipitable Water (PW)", value=40.0)
    submitted = st.form_submit_button("🔍 Prediksi")

# === 5. Proses Prediksi ===
if submitted:
    # --- Klasifikasi ---
    X_input_cls = pd.DataFrame([{
        'LI': LI,
        'SWEAT': SWEAT,
        'KI': KI,
        'TTI': TTI,
        'CAPE': CAPE,
        'SI': SI,
        'PW': PW
    }])[fitur_model]

    prob = model_cls.predict_proba(X_input_cls)[0, 1]
    klasifikasi = "⚡ Petir" if prob >= 0.5 else "✅ Non-Petir"

    st.metric("Probabilitas Petir", f"{prob * 100:.1f}%")
    st.success(f"Hasil Klasifikasi: {klasifikasi}")

   # --- Regresi Jika Terjadi Petir ---
if prob >= 0.5:
    X_input_reg = pd.DataFrame([{
        'LI': LI,
        'SWEAT': SWEAT,
        'KI': KI,
        'TTI': TTI,
        'CAPE': CAPE,
        'SI': SI,
        'PW': PW
    }])[fitur_model_reg]

    # Langsung prediksi (tanpa transformasi log)
    pred_reg = model_reg.predict(X_input_reg)
    pred_reg = np.clip(pred_reg, a_min=0, a_max=10000)  # batas maksimum opsional

    try:
        st.metric("Estimasi Jumlah Sambaran CG", f"{int(pred_reg[0]):,} sambaran")
        st.info("Prediksi jumlah sambaran CG hanya ditampilkan jika petir terdeteksi.")
    except OverflowError:
        st.error("❗ Prediksi jumlah sambaran terlalu besar untuk ditampilkan.")
