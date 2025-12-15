import streamlit as st
import pandas as pd
import sqlite3
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# ===============================
# DATABASE
# ===============================
conn = sqlite3.connect("stres.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS riwayat_stres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jam_tidur INTEGER,
    beban_tugas TEXT,
    emosi TEXT,
    hasil TEXT
)
""")
conn.commit()

# ===============================
# MACHINE LEARNING
# ===============================
data = pd.read_csv("data_stres.csv")
X = data[['jam_tidur', 'beban_tugas', 'emosi']]
y = data['stres']

model = GaussianNB()
model.fit(X, y)

# ===============================
# USER INTERFACE
# ===============================
st.set_page_config(page_title="Sistem Cerdas Stres")

st.title("🧠 Sistem Cerdas Prediksi Stres Mahasiswa")
st.write("Machine Learning + Database + Visualisasi")

# ===============================
# INPUT USER
# ===============================
st.header("🔍 Input Data")

jam_tidur = st.slider("Jam tidur per hari", 0, 10, 6)
beban = st.selectbox("Beban tugas", ["Ringan", "Sedang", "Banyak"])
emosi = st.selectbox("Tekanan emosional", ["Rendah", "Sedang", "Tinggi"])

beban_val = {"Ringan":1, "Sedang":2, "Banyak":3}[beban]
emosi_val = {"Rendah":1, "Sedang":2, "Tinggi":3}[emosi]

if st.button("Prediksi & Simpan"):
    hasil = model.predict([[jam_tidur, beban_val, emosi_val]])[0]

    cursor.execute(
        "INSERT INTO riwayat_stres (jam_tidur, beban_tugas, emosi, hasil) VALUES (?, ?, ?, ?)",
        (jam_tidur, beban, emosi, hasil)
    )
    conn.commit()

    st.success(f"Hasil Prediksi: {hasil}")

# ===============================
# RIWAYAT
# ===============================
st.header("📋 Riwayat Diagnosa")
df = pd.read_sql("SELECT * FROM riwayat_stres", conn)
st.dataframe(df)

# ===============================
# VISUALISASI
# ===============================
if not df.empty:
    st.header("📊 Visualisasi & Analisis")

    # Grafik jumlah stres
    fig1, ax1 = plt.subplots()
    df['hasil'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Distribusi Tingkat Stres")
    ax1.set_xlabel("Tingkat Stres")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

    # Analisis jam tidur rata-rata
    avg_sleep = df.groupby('hasil')['jam_tidur'].mean()

    fig2, ax2 = plt.subplots()
    avg_sleep.plot(kind='bar', ax=ax2)
    ax2.set_title("Rata-rata Jam Tidur per Tingkat Stres")
    ax2.set_xlabel("Tingkat Stres")
    ax2.set_ylabel("Jam Tidur")
    st.pyplot(fig2)

    # Insight otomatis
    st.subheader("🧠 Insight Sistem")
    if avg_sleep.min() < 6:
        st.warning("Mahasiswa dengan stres tinggi cenderung memiliki jam tidur rendah.")
    else:
        st.info("Pola tidur relatif baik.")
