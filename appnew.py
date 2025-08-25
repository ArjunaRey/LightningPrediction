# app.py
import streamlit as st
import pandas as pd
import joblib

# === Load model ===
clf_model = joblib.load("xgb_classifier.joblib")
reg_model = joblib.load("xgb_regressor.joblib")

# Daftar prediktor sesuai dataset (hapus ws10)
predictors = [
    "t_500", "t_700", "t_850", "z_1000", "r_500", "r_700", "r_850", "w_700",
    "cape", "msl", "tcc", "d2m", "tcrw", "u500", "u850", "v500", "v850",
    "sp", "u10", "v10", "t2m"
]

st.set_page_config(page_title="Prediksi Kejadian Petir", layout="wide")

st.title("⚡ Prediksi Kejadian dan Jumlah Sambaran Petir")
st.write("Masukkan parameter cuaca sesuai variabel prediktor.")

# === Form Input ===
input_data = {}
cols = st.columns(3)  # bagi jadi 3 kolom biar rapi

for i, feature in enumerate(predictors):
    with cols[i % 3]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

# === Prediksi ===
if st.button("🔍 Prediksi"):
    df_input = pd.DataFrame([input_data])

    # Samakan urutan kolom dengan model
    df_input = df_input[clf_model.feature_names_in_]

    # Klasifikasi (kejadian petir)
    pred_class = clf_model.predict(df_input)[0]
    prob_class = clf_model.predict_proba(df_input)[0][1]

    # Regresi (jumlah sambaran)
    if pred_class == 1:
        pred_reg = reg_model.predict(df_input)[0]
    else:
        pred_reg = 0

    # === Output ===
    st.subheader("Hasil Prediksi")
    if pred_class == 1:
        st.success(f"⚡ Petir DIPREDIKSI terjadi (Probabilitas {prob_class:.2%})")
        st.info(f"Perkiraan jumlah sambaran petir: **{pred_reg:.0f} kali**")
    else:
        st.warning(f"☀️ Tidak ada petir (Probabilitas {prob_class:.2%})")
