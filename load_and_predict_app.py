import streamlit as st
import pandas as pd
import joblib

# Load model
clf = joblib.load("model_klasifikasi_petir.pkl")
reg_model = joblib.load("model_regresi_petir.pkl")

# Upload data
st.title("Prediksi Petir dan Jumlah Sambaran CG")
uploaded = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year

    fitur = [col for col in df.columns if col not in ["timestamp", "label_petir_biner", "count_total_cg", "year", "season", "hour", "month"]]
    X = df[fitur]

    # Prediksi klasifikasi
    y_pred = clf.predict(X)
    df["prediksi_petir"] = y_pred

    # Prediksi regresi jika petir diprediksi terjadi
    if (df["prediksi_petir"] == 1).any():
        df_reg = df[df["prediksi_petir"] == 1]
        X_reg = df_reg[fitur]
        y_reg_pred = reg_model.predict(X_reg)
        df.loc[df["prediksi_petir"] == 1, "prediksi_jumlah_cg"] = y_reg_pred

    st.success("✅ Prediksi selesai.")
    st.dataframe(df[["timestamp", "prediksi_petir", "prediksi_jumlah_cg"]].head(20))
    st.download_button("📥 Download hasil", data=df.to_csv(index=False), file_name="hasil_prediksi.csv", mime="text/csv")
