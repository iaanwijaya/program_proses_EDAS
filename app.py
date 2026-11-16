import io
import numpy as np
import pandas as pd
import streamlit as st

# 1. KONFIGURASI HALAMAN & CUSTOM CSS
st.set_page_config(
    page_title="SPK Pemilihan Kantin - Metode EDAS",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk mempercantik tampilan card, metric, dan tabel
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 2. DATA DEFAULT & FUNGSI PERHITUNGAN METODE EDAS
@st.cache_data
def load_default_data():
    """Membuat dataset fallback (sesuai data matriks keputusan 7 kantin)

    jika file CSV belum diunggah atau tidak ditemukan di folder lokal.
    """
    data = {
        "Kantin": [
            "Kantin Fasilkom",
            "Kantin FH",
            "Kantin S1 FE",
            "Kantin D3 FE",
            "Kantin MM",
            "Kantin Pantai",
            "Kantin Polsri",
        ],
        "Harga": [
            14500.00,
            16250.00,
            17500.00,
            15000.00,
            19166.67,
            15000.00,
            14166.67,
        ],
        "Waktu Tunggu": [8.50, 8.75, 15.00, 5.00, 6.67, 10.00, 11.67],
        "Kualitas": [3.40, 3.50, 4.33, 4.00, 4.17, 3.00, 4.17],
        "Kesesuaian Porsi": [3.10, 3.75, 4.67, 5.00, 4.17, 3.33, 3.83],
        "Kebersihan Meja": [11.50, 7.50, 18.33, 5.00, 9.17, 6.67, 11.67],
        "Kursi Jam Sibuk": [52.00, 45.00, 46.67, 40.00, 56.67, 33.33, 43.33],
        "Fasilitas": [2.60, 3.50, 2.83, 3.00, 4.00, 2.33, 3.33],
        "Pelayanan": [3.60, 4.25, 4.50, 4.00, 4.67, 3.33, 3.83],
    }
    return pd.DataFrame(data)


def map_columns(df):
    """Mencocokkan nama kolom dari dataset CSV (baik nama pendek maupun panjang dengan satuan)."""
    col_mapping = {}
    target_keywords = {
        "Kantin": ["kantin", "alternatif", "nama kantin"],
        "Harga": ["harga"],
        "Waktu Tunggu": ["waktu tunggu", "tunggu"],
        "Kualitas": ["kualitas"],
        "Kesesuaian Porsi": ["kesesuaian porsi", "porsi"],
        "Kebersihan Meja": ["kebersihan meja", "kebersihan"],
        "Kursi Jam Sibuk": ["kursi jam sibuk", "kursi"],
        "Fasilitas": ["fasilitas"],
        "Pelayanan": ["pelayanan"],
    }

    for standard_name, keywords in target_keywords.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(kw in col_lower for kw in keywords):
                col_mapping[col] = standard_name
                break

    df_renamed = df.rename(columns=col_mapping)
    return df_renamed


def hitung_edas(df_raw, bobot_dict, sifat_dict):
    """Menjalankan seluruh tahapan matematis Metode EDAS (Step 1 sampai Step 7)."""
    kolom_kriteria = list(sifat_dict.keys())

    # STEP 1: Agregasi Matriks Keputusan (X_ij) berdasarkan rata-rata per Kantin
    matriks_keputusan = df_raw.groupby("Kantin")[kolom_kriteria].mean()

    # STEP 2: Average Solution (AV_j)
    av = matriks_keputusan.mean(axis=0)

    # STEP 3 & 4: Matriks PDA & NDA
    pda = pd.DataFrame(0.0, index=matriks_keputusan.index, columns=kolom_kriteria)
    nda = pd.DataFrame(0.0, index=matriks_keputusan.index, columns=kolom_kriteria)

    for col in kolom_kriteria:
        av_j = av[col]
        x_j = matriks_keputusan[col]

        if sifat_dict[col] == "Benefit":
            pda[col] = np.maximum(0, (x_j - av_j) / av_j)
            nda[col] = np.maximum(0, (av_j - x_j) / av_j)
        else:  # Cost
            pda[col] = np.maximum(0, (av_j - x_j) / av_j)
            nda[col] = np.maximum(0, (x_j - av_j) / av_j)

    # STEP 5: Jumlah Terbobot SP & SN
    bobot_array = np.array([bobot_dict[col] for col in kolom_kriteria])
    sp = pda.dot(bobot_array)
    sn = nda.dot(bobot_array)

    # STEP 6: Normalisasi NSP & NSN
    max_sp = sp.max() if sp.max() != 0 else 1
    max_sn = sn.max() if sn.max() != 0 else 1

    nsp = sp / max_sp
    nsn = 1 - (sn / max_sn)

    # STEP 7: Skor Akhir (AS) & Perangkingan
    as_score = (nsp + nsn) / 2

    df_hasil = pd.DataFrame(
        {
            "SP": sp,
            "SN": sn,
            "NSP": nsp,
            "NSN": nsn,
            "Skor Akhir (AS)": as_score,
        }
    )

    df_hasil["Ranking"] = (
        df_hasil["Skor Akhir (AS)"].rank(ascending=False, method="min").astype(int)
    )
    df_hasil = df_hasil.sort_values(by="Ranking")

    return {
        "matriks_keputusan": matriks_keputusan,
        "av": av,
        "pda": pda,
        "nda": nda,
        "hasil_akhir": df_hasil,
    }


# 3. SIDEBAR: UPLOAD DATASET & PENGATURAN BOBOT DINAMIS
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3448/3448609.png", width=65)
    st.title("Panel Kontrol EDAS")
    st.markdown("---")

    # A. Sumber Data
    st.subheader("1. Sumber Dataset")
    uploaded_file = st.file_uploader(
        "Unggah file CSV kuesioner",
        type=["csv"],
        help="Unggah file data_kuesioner_kantin.csv milik Anda.",
    )

    # Load data (prioritas: file upload -> file lokal -> data default)
    df_input = None
    sumber_info = ""
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        sumber_info = f"File diunggah: **{uploaded_file.name}** ({len(df_input)} baris data)"
    else:
        try:
            df_input = pd.read_csv("data_kuesioner_kantin.csv")
            sumber_info = f"File lokal: **data_kuesioner_kantin.csv** ({len(df_input)} baris data)"
        except FileNotFoundError:
            df_input = load_default_data()
            sumber_info = "Menggunakan **Data Bawaan Sistem** (7 Kantin)"

    df_clean = map_columns(df_input)

    st.markdown("---")

    # B. Pengaturan Bobot Kriteria
    st.subheader("2. Bobot Kriteria (w)")
    st.caption(
        "Geser slider untuk menyesuaikan tingkat kepentingan. Sistem otomatis menormalisasi total bobot menjadi 1.00 (100%)."
    )

    if st.button("🔄 Reset ke Bobot Awal", use_container_width=True):
        st.session_state["w_harga"] = 0.15
        st.session_state["w_tunggu"] = 0.10
        st.session_state["w_kualitas"] = 0.20
        st.session_state["w_porsi"] = 0.15
        st.session_state["w_kebersihan"] = 0.10
        st.session_state["w_kursi"] = 0.10
        st.session_state["w_fasilitas"] = 0.10
        st.session_state["w_pelayanan"] = 0.10

    raw_weights = {
        "Harga": st.slider("Harga (Cost)", 0.0, 1.0, st.session_state.get("w_harga", 0.15), 0.05, key="w_harga"),
        "Waktu Tunggu": st.slider("Waktu Tunggu (Cost)", 0.0, 1.0, st.session_state.get("w_tunggu", 0.10), 0.05, key="w_tunggu"),
        "Kualitas": st.slider("Kualitas Makanan (Benefit)", 0.0, 1.0, st.session_state.get("w_kualitas", 0.20), 0.05, key="w_kualitas"),
        "Kesesuaian Porsi": st.slider("Kesesuaian Porsi (Benefit)", 0.0, 1.0, st.session_state.get("w_porsi", 0.15), 0.05, key="w_porsi"),
        "Kebersihan Meja": st.slider("Kebersihan Meja (Cost)", 0.0, 1.0, st.session_state.get("w_kebersihan", 0.10), 0.05, key="w_kebersihan"),
        "Kursi Jam Sibuk": st.slider("Kursi Jam Sibuk (Benefit)", 0.0, 1.0, st.session_state.get("w_kursi", 0.10), 0.05, key="w_kursi"),
        "Fasilitas": st.slider("Fasilitas (Benefit)", 0.0, 1.0, st.session_state.get("w_fasilitas", 0.10), 0.05, key="w_fasilitas"),
        "Pelayanan": st.slider("Pelayanan (Benefit)", 0.0, 1.0, st.session_state.get("w_pelayanan", 0.10), 0.05, key="w_pelayanan"),
    }

    total_raw = sum(raw_weights.values())
    if total_raw == 0:
        st.error("Total bobot tidak boleh 0!")
        st.stop()

    # Normalisasi bobot agar selalu berjumlah 1.0
    bobot_normal = {k: v / total_raw for k, v in raw_weights.items()}

    if not np.isclose(total_raw, 1.0):
        st.warning(f"Total bobot input: **{total_raw:.2f}**. Otomatis dinormalisasi ke **1.00**.")
    else:
        st.success(f"Total Bobot Valid: **{total_raw:.2f} (100%)**")

# 4. EKSEKUSI PERHITUNGAN & HALAMAN UTAMA (DASHBOARD)
sifat_kriteria = {
    "Harga": "Cost",
    "Waktu Tunggu": "Cost",
    "Kualitas": "Benefit",
    "Kesesuaian Porsi": "Benefit",
    "Kebersihan Meja": "Cost",
    "Kursi Jam Sibuk": "Benefit",
    "Fasilitas": "Benefit",
    "Pelayanan": "Benefit",
}

# Validasi kolom sebelum proses
kolom_wajib = ["Kantin"] + list(sifat_kriteria.keys())
kolom_hilang = [col for col in kolom_wajib if col not in df_clean.columns]

if kolom_hilang:
    st.error(f"Kolom berikut tidak ditemukan di dataset Anda: {', '.join(kolom_hilang)}")
    st.info("Pastikan file CSV memiliki kolom: Kantin, Harga, Waktu Tunggu, Kualitas, Kesesuaian Porsi, Kebersihan Meja, Kursi Jam Sibuk, Fasilitas, dan Pelayanan.")
    st.stop()

# Jalankan kalkulasi EDAS
edas_res = hitung_edas(df_clean, bobot_normal, sifat_kriteria)
df_hasil = edas_res["hasil_akhir"]

# Header Utama
st.markdown('<div class="main-header">🍽️ Sistem Pendukung Keputusan Pemilihan Kantin</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Evaluasi Multi-Kriteria Menggunakan Metode <b>EDAS</b> (<i>Evaluation based on Distance from Average Solution</i>) • {sumber_info}</div>',
    unsafe_allow_html=True,
)

# Highlight KPI Cards (Top 3 Kantin Terbaik)
top_kantin = df_hasil.index[0]
top_score = df_hasil.iloc[0]["Skor Akhir (AS)"]

runner_up = df_hasil.index[1]
runner_score = df_hasil.iloc[1]["Skor Akhir (AS)"]

third_place = df_hasil.index[2]
third_score = df_hasil.iloc[2]["Skor Akhir (AS)"]

col1, col2, col3, col4 = st.columns([1.3, 1.1, 1.1, 1])

with col1:
    st.metric(
        label="🏆 Rekomendasi Terbaik (Rank #1)",
        value=top_kantin,
        delta=f"Skor AS: {top_score:.4f}",
    )
with col2:
    st.metric(
        label="🥈 Peringkat #2",
        value=runner_up,
        delta=f"Skor AS: {runner_score:.4f}",
        delta_color="off",
    )
with col3:
    st.metric(
        label="🥉 Peringkat #3",
        value=third_place,
        delta=f"Skor AS: {third_score:.4f}",
        delta_color="off",
    )
with col4:
    st.metric(
        label="📊 Total Alternatif & Kriteria",
        value=f"{len(df_hasil)} Kantin",
        delta="8 Kriteria (5 Benefit, 3 Cost)",
        delta_color="off",
    )

st.markdown("---")

# 5. TABS INTERAKTIF: HASIL PERANGKINGAN & TRANSPARANSI STEP 1 - 7
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏆 Hasil Perangkingan & Grafik",
        "📋 Step 1 & 2: Matriks Keputusan & AV",
        "📐 Step 3 & 4: Matriks PDA & NDA",
        "📑 Data Mentah Kuesioner",
    ]
)

# --- TAB 1: HASIL AKHIR & GRAFIK ---
with tab1:
    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        st.subheader("Tabel Perangkingan Akhir (Step 5 - 7)")
        # Format tampilan angka agar rapi 4 desimal
        styled_hasil = df_hasil.style.format(
            {
                "SP": "{:.4f}",
                "SN": "{:.4f}",
                "NSP": "{:.4f}",
                "NSN": "{:.4f}",
                "Skor Akhir (AS)": "{:.4f}",
                "Ranking": "#{}",
            }
        ).background_gradient(subset=["Skor Akhir (AS)"], cmap="Blues")

        st.dataframe(styled_hasil, use_container_width=True, height=300)

        # Tombol Download Hasil Excel / CSV
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_hasil.to_excel(writer, sheet_name="Hasil_Ranking_EDAS")
            edas_res["matriks_keputusan"].to_excel(writer, sheet_name="Matriks_Keputusan")
            edas_res["pda"].to_excel(writer, sheet_name="Matriks_PDA")
            edas_res["nda"].to_excel(writer, sheet_name="Matriks_NDA")

        st.download_button(
            label="📥 Download Laporan Lengkap EDAS (.xlsx)",
            data=buffer.getvalue(),
            file_name="Hasil_Analisis_EDAS_Kantin.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_chart:
        st.subheader("Visualisasi Skor Akhir (AS)")
        chart_data = df_hasil[["Skor Akhir (AS)"]].sort_values(by="Skor Akhir (AS)", ascending=True)
        st.bar_chart(chart_data, horizontal=True, color="#2563EB", height=320)

    # Perbandingan NSP vs NSN
    st.subheader("Perbandingan Nilai Normalisasi: Keunggulan (NSP) vs Kelemahan (NSN)")
    st.caption(
        "**NSP** (semakin tinggi semakin unggul di atas rata-rata) • **NSN** (semakin mendekati 1 semakin minim kelemahan di bawah rata-rata)."
    )
    st.bar_chart(df_hasil[["NSP", "NSN"]], height=280)

# --- TAB 2: MATRIKS KEPUTUSAN & AV ---
with tab2:
    st.subheader("Step 1: Matriks Keputusan (Rata-rata Penilaian Responden)")
    st.caption("Nilai rata-rata untuk masing-masing kantin pada 8 kriteria penilaian.")
    st.dataframe(
        edas_res["matriks_keputusan"].style.format("{:.2f}"),
        use_container_width=True,
    )

    st.subheader("Step 2: Informasi Kriteria, Bobot Aktif, & Average Solution (AV)")
    df_info_kriteria = pd.DataFrame(
        {
            "Sifat Kriteria": sifat_kriteria,
            "Bobot Aktif (w)": bobot_normal,
            "Average Solution (AV)": edas_res["av"],
        }
    ).T
    st.dataframe(df_info_kriteria, use_container_width=True)

# --- TAB 3: MATRIKS PDA & NDA ---
with tab3:
    col_pda, col_nda = st.columns(2)
    with col_pda:
        st.subheader("Step 3: Matriks PDA")
        st.caption("*Positive Distance from Average* (Jarak Keunggulan)")
        st.dataframe(
            edas_res["pda"].style.format("{:.4f}").background_gradient(cmap="Greens"),
            use_container_width=True,
        )

    with col_nda:
        st.subheader("Step 4: Matriks NDA")
        st.caption("*Negative Distance from Average* (Jarak Kelemahan)")
        st.dataframe(
            edas_res["nda"].style.format("{:.4f}").background_gradient(cmap="Reds"),
            use_container_width=True,
        )

# --- TAB 4: DATA MENTAH ---
with tab4:
    st.subheader("Dataset Kuesioner yang Digunakan")
    st.write(f"Menampilkan **{len(df_input)}** baris data dari sumber dataset:")
    st.dataframe(df_input, use_container_width=True)