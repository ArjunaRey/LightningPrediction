import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === 1. Load models ===
clf_model = joblib.load("xgb_classifier.joblib")
reg_model = joblib.load("xgb_regressor.joblib")

# === 2. Ambil urutan fitur dari model klasifikasi ===
try:
    predictors = list(clf_model.feature_names_in_)
except AttributeError:
    predictors = clf_model.get_booster().feature_names

# === 3. Load dataset untuk ambil rata-rata (pastikan file sama dengan training) ===
df = pd.read_csv("dataset-clean-capped.csv")
feature_means = df[predictors].mean()

# === 4. Streamlit UI ===
st.title("Aplikasi Prediksi Petir CG (Dua Tahap)")
st.markdown("Masukkan nilai parameter atmosfer berikut:")

# Input data dalam layout 3 kolom, default = rata-rata dataset
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(predictors):
    default_val = float(feature_means.get(feature, 0.0))  # fallback jika kolom hilang
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            value=default_val,
            step=0.1,
            format="%.2f"
        )

# Buat DataFrame sesuai urutan fitur model
df_input = pd.DataFrame([input_data])

# === 5. Pastikan input sesuai dengan model ===
# Tambahkan kolom yang hilang
for col in predictors:
    if col not in df_input.columns:
        df_input[col] = 0.0

# Hapus kolom ekstra
extra_cols = [c for c in df_input.columns if c not in predictors]
if extra_cols:
    df_input.drop(columns=extra_cols, inplace=True)

# Susun ulang kolom sesuai urutan model
df_input = df_input[predictors].astype(float)

# Debugging: tampilkan agar tahu kalau mismatch
st.write("🔍 Model features:", predictors)
st.write("🔍 Input features:", list(df_input.columns))

# === 6. Prediksi ===
if st.button("Prediksi Petir"):
    try:
        # Tahap 1: Klasifikasi ada/tidaknya petir
        pred_class = clf_model.predict(df_input)[0]
        prob_class = clf_model.predict_proba(df_input)[0][1]

        st.subheader("Hasil Klasifikasi")
        if pred_class == 1:
            st.success(f"**Petir terdeteksi** (Probabilitas: {prob_class:.2f})")

            # Tahap 2: Prediksi jumlah sambaran
            pred_count = reg_model.predict(df_input)[0]
            pred_count = np.maximum(pred_count, 0)
            st.subheader("Prediksi Jumlah Sambaran CG")
            st.info(f"Perkiraan jumlah sambaran: {pred_count:.2f} kali")
        else:
            st.error(f"Tidak terdeteksi petir (Probabilitas: {prob_class:.2f})")

    except ValueError as e:
        st.error("⚠️ Terjadi mismatch fitur antara input dan model.")
        st.code(str(e))
