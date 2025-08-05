# === 1. Import Library ===
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === 2. CSS untuk latar putih & teks bersih ===
# === SETTING TAMPILAN PUTIH ===

# === 2. Load Model Klasifikasi dan Dua Regresi (Log + Linear) ===
model_cls, fitur_model = joblib.load("model_klasifikasi_petir4.pkl")
model_log, fitur_model_log = joblib.load("model_regresi_log.pkl")
model_lin, fitur_model_lin = joblib.load("model_regresi_linear.pkl")

with st.sidebar:
    st.image("logo_stmkg.png", width=140)
    st.markdown("#### Arjuna Reynaldi")
    st.markdown("*STMKG 2025*")
    st.markdown("arjunareynaldi58@gmail.com")
    st.markdown("---")
    st.markdown("📘 *Aplikasi ini memprediksi petir dan estimasi sambaran CG berdasarkan parameter atmosfer.*")
    st.markdown("🧠 Model: XGBClassifier & XGBRegressor")
    with st.expander("ℹ️ Cara Penggunaan"):
        st.markdown("""
        - Isi parameter atmosfer via slider
        - Klik **Prediksi**
        - Jika petir terdeteksi, hasil estimasi sambaran muncul
        """)


# === 3. Streamlit Interface ===
st.set_page_config(page_title="Prediksi Petir & Jumlah Sambaran CG", layout="centered")
st.title("🌩️ Prediksi Kejadian dan Estimasi Jumlah Sambaran CG")
st.markdown("Masukkan parameter atmosfer berikut:")

# === 4. Form Input User ===
# === FORM INPUT ===
with st.form("form_input"):
    st.subheader("Masukkan Parameter Atmosfer:")

    LI = st.number_input("Lifted Index (LI)", min_value=-10.0, max_value=10.0, value=-2.0, step=0.01)
    SWEAT = st.number_input("SWEAT Index", min_value=100.0, max_value=300.0, value=200.0, step=0.01)
    KI = st.number_input("K Index", min_value=10.0, max_value=50.0, value=30.0, step=0.01)
    TTI = st.number_input("Total Totals Index (TTI)", min_value=20.0, max_value=50.0, value=48.0, step=0.01)
    CAPE = st.number_input("CAPE (J/kg)", min_value=0.0, max_value=3000.0, value=1000.0, step=0.01)
    SI = st.number_input("Showalter Index (SI)", min_value=-5.0, max_value=5.0, value=1.0, step=0.01)
    PW = st.number_input("Precipitable Water (PW)", min_value=30.0, max_value=80.0, value=40.0, step=0.01)

    submit = st.form_submit_button("Prediksi")


# === 5. Proses Prediksi ===
if submit:
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
    klasifikasi = "⚡ Diprediksi akan terjadi sambaran CG dalam 12 jam ke depan" if prob >= 0.5 else "✅ Diprediksi tidak akan terjadi sambaran CG dalam 12 jam ke depan"

    st.metric("Probabilitas Petir", f"{prob:.0%}")
    st.success(f"Hasil Klasifikasi: {klasifikasi}")

    # --- Regresi Jika Terjadi Petir ---
    if prob >= 0.5:
        # Input untuk kedua model regresi
        X_input_reg_log = pd.DataFrame([{
            'LI': LI,
            'SWEAT': SWEAT,
            'KI': KI,
            'TTI': TTI,
            'CAPE': CAPE,
            'SI': SI,
            'PW': PW
        }])[fitur_model_log]

        X_input_reg_lin = pd.DataFrame([{
            'LI': LI,
            'SWEAT': SWEAT,
            'KI': KI,
            'TTI': TTI,
            'CAPE': CAPE,
            'SI': SI,
            'PW': PW
        }])[fitur_model_lin]

        # Prediksi dari model log & linear
        pred_log = np.expm1(model_log.predict(X_input_reg_log))[0]
        pred_lin = model_lin.predict(X_input_reg_lin)[0]

        # Ensemble rata-rata tertimbang
        alpha = 0.7
        final_pred = alpha * pred_log + (1 - alpha) * pred_lin
        final_pred = np.clip(final_pred, 0, 10000)

        try:
            st.metric("Estimasi Jumlah Sambaran", f"{int(final_pred):,} sambaran")
            #st.info(f"Prediksi gabungan model log (α={alpha}) dan linear.")
        except OverflowError:
            st.error("❗ Prediksi jumlah sambaran terlalu besar untuk ditampilkan.")
    else:
        st.warning("Tidak ada prediksi petir, sehingga estimasi jumlah sambaran tidak ditampilkan.")
