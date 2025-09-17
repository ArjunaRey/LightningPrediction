import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === 1. Sidebar Info ===
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

# === 2. Load models ===
clf_model = joblib.load("xgb_classifier.joblib")
reg_model = joblib.load("xgb_regressor.joblib")

# === 3. Ambil urutan fitur dari model klasifikasi ===
try:
    predictors = list(clf_model.feature_names_in_)
except AttributeError:
    predictors = clf_model.get_booster().feature_names

# === 4. Load dataset untuk ambil rata-rata ===
df = pd.read_csv("dataset-clean-capped.csv")
feature_means = df[predictors].mean()

# === 5. Streamlit UI utama ===
st.title("Aplikasi Prediksi Kejadian dan Jumlah Sambaran CG ")
st.markdown("Masukkan data reanalisis ERA5 berikut:")

# Input data dalam layout 3 kolom
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(predictors):
    default_val = float(feature_means.get(feature, 0.0))
    min_val = -100.0  # batas minimal semua fitur
    max_val = 120000.0     # batas maksimal semua fitur

    # Pastikan default_val tidak di bawah min atau di atas max
    default_val = max(default_val, min_val)
    default_val = min(default_val, max_val)

    with cols[i % 3]:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            value=default_val,
            min_value=min_val,
            max_value=max_val,
            step=0.1,
            format="%.4f"
        )

# Buat DataFrame sesuai input user
df_input = pd.DataFrame([input_data])

# Pastikan input sesuai model
df_input = df_input.reindex(columns=predictors, fill_value=0).astype(float)

# === 6. Prediksi ===
if st.button("Prediksi Petir"):
    try:
        # Tahap 1: Klasifikasi ada/tidaknya petir
        pred_class = clf_model.predict(df_input)[0]
        proba = clf_model.predict_proba(df_input)
        prob_class = proba[0][1] if proba.shape[1] > 1 else proba[0][0]

        if pred_class == 1:
            st.subheader("Hasil Klasifikasi")
            st.success(f"**Diprediksi akan terjadi sambaran dalam 3 jam ke depan** (Probabilitas: {prob_class:.2f})")

            # Tahap 2: Prediksi jumlah sambaran
            pred_count = reg_model.predict(df_input)[0]
            pred_count = int(np.round(np.maximum(pred_count, 0)))
            st.subheader("Prediksi Jumlah Sambaran CG")
            st.info(f"Perkiraan jumlah sambaran: {pred_count} kali")

        else:
            st.subheader("Hasil Klasifikasi")
            st.error(f"Diprediksi tidak akan terjadi sambaran dalam 3 jam ke depan (Probabilitas: {prob_class:.2f})")

    except ValueError:
        st.error("⚠️ Terjadi mismatch fitur antara input dan model.")
