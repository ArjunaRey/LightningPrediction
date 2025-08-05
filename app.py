import pandas as pd
import numpy as np
import joblib
import streamlit as st

# === CONFIG PAGE ===
st.set_page_config(page_title="Prediksi Petir CG", layout="wide", page_icon="🌩️")

# === CSS CUSTOM: Full Override ===
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        background-color: white !important;
        color: #111 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Form styling */
    .stSlider > div, .stNumberInput input {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, .stMarkdown, label {
        color: #111 !important;
    }

    .stButton > button {
        background-color: #0c66f5 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
    }

    .stAlert {
        color: #111 !important;
    }

    .st-bz, .st-c2 {
        color: #111 !important;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR: Logo + Author ===
with st.sidebar:
    st.image("a97ab832eabb.png", width=140)
    st.markdown("""
        <div style='color:white; font-size:18px; font-weight:bold; margin-top:10px;'>
            Arjuna Reynaldi
        </div>
        <div style='color:white; font-size:14px;'>
            STMKG 2025
        </div>
    """, unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
    <h1 style='font-size:32px; font-weight:bold;'>🌩️ Prediksi Kejadian Petir & Estimasi Sambaran CG</h1>
    <p style='font-size:17px;'>Masukkan parameter atmosfer berikut:</p>
""", unsafe_allow_html=True)

# === FORM INPUT ===
with st.form("form_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        LI = st.slider("Lifted Index (LI)", -10.0, 10.0, -2.0, step=0.1)
        SWEAT = st.slider("SWEAT Index", 100.0, 600.0, 200.0, step=1.0)
    with col2:
        KI = st.slider("K Index", 10.0, 50.0, 30.0, step=1.0)
        TTI = st.slider("Total Totals Index (TTI)", 20.0, 70.0, 48.0, step=1.0)
    with col3:
        CAPE = st.slider("CAPE (J/kg)", 0.0, 5000.0, 1000.0, step=10.0)
        SI = st.slider("Showalter Index (SI)", -10.0, 10.0, 1.0, step=0.1)
    PW = st.slider("Precipitable Water (PW)", 10.0, 80.0, 40.0, step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi")

# === 2. Load Model Klasifikasi dan Dua Regresi (Log + Linear) ===
model_cls, fitur_model = joblib.load("model_klasifikasi_petir4.pkl")
model_log, fitur_model_log = joblib.load("model_regresi_log.pkl")
model_lin, fitur_model_lin = joblib.load("model_regresi_linear.pkl")

# === PREDIKSI ===
if submitted:
    # Klasifikasi
    X_input_cls = pd.DataFrame([{
        'LI': LI, 'SWEAT': SWEAT, 'KI': KI, 'TTI': TTI,
        'CAPE': CAPE, 'SI': SI, 'PW': PW
    }])[fitur_model]

    prob = model_cls.predict_proba(X_input_cls)[0, 1]
    klasifikasi = "🔦 Petir" if prob >= 0.5 else "✅ Non-Petir"

    st.markdown(f"""
        <div style='margin-top:20px; font-size:22px; font-weight:bold;'>
            Probabilitas Petir: {prob:.0%}
        </div>
        <div style="margin-top:10px; padding:15px; background-color:#e6f4ea; border-left:6px solid #4CAF50; border-radius:8px; font-size:18px;">
            Hasil Klasifikasi: {klasifikasi}
        </div>
    """, unsafe_allow_html=True)

    # Regresi jika Petir
    if prob >= 0.5:
        X_input_reg = pd.DataFrame([{
            'LI': LI, 'SWEAT': SWEAT, 'KI': KI, 'TTI': TTI,
            'CAPE': CAPE, 'SI': SI, 'PW': PW
        }])[fitur_model_reg]

        pred_reg = model_reg.predict(X_input_reg)
        pred_reg = np.clip(pred_reg, a_min=0, a_max=10000)

        st.markdown(f"""
            <div style='margin-top:20px; font-size:22px; font-weight:bold;'>
                Estimasi Jumlah Sambaran CG: {int(pred_reg[0]):,} sambaran
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Tidak ada prediksi petir, sehingga estimasi jumlah sambaran tidak ditampilkan.")
