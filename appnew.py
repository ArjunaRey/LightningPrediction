import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === 1. Load models ===
clf_model = joblib.load("xgb_classifier.joblib")
reg_model = joblib.load("xgb_regressor.joblib")

# === 2. Ambil urutan fitur dari model klasifikasi ===
predictors = list(clf_model.feature_names_in_)

# === 3. Load dataset untuk ambil rata-rata (pastikan file sama dengan training)
df = pd.read_excel("dataset_clean_capped.xlsx")
feature_means = df[predictors].mean()

# === 4. Streamlit UI ===
st.title("Aplikasi Prediksi Petir CG (Dua Tahap)")
st.markdown("Masukkan nilai parameter atmosfer berikut:")

# Input data dalam layout 3 kolom, default = rata-rata dataset
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(predictors):
    default_val = float(feature_means[feature])
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            label=f"{feature}", 
            value=default_val, 
            step=0.1,
            format="%.2f"
        )

# Buat DataFrame sesuai urutan fitur model
df_input = pd.DataFrame([input_data])[predictors]

# === 5. Prediksi ===
if st.button("Prediksi Petir"):
    # Tahap 1: Klasifikasi ada/tidaknya petir
    pred_class = clf_model.predict(df_input)[0]
    prob_class = clf_model.predict_proba(df_input)[0][1]  # probabilitas petir

    st.subheader("Hasil Klasifikasi")
    if pred_class == 1:
        st.success(f"**Petir terdeteksi** (Probabilitas: {prob_class:.2f})")
        
        # Tahap 2: Prediksi jumlah sambaran
        pred_count = reg_model.predict(df_input)[0]
        pred_count = np.maximum(pred_count, 0)  # hindari prediksi negatif
        st.subheader("Prediksi Jumlah Sambaran CG")
        st.info(f"Perkiraan jumlah sambaran: {pred_count:.2f} kali")
    else:
        st.error(f"Tidak terdeteksi petir (Probabilitas: {prob_class:.2f})")
