import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from pathlib import Path
from collections import Counter

def load_sample_csv():
    """Mengambil contoh dataset CSV untuk di-download oleh user."""
    sample_path = Path("assets/datasetTest.csv")   # Pastikan file di sini
    if not sample_path.exists():
        return None

    with open(sample_path, "rb") as f:
        return f.read()

# ========== CONFIG STREAMLIT ==========
st.set_page_config(
    page_title="Sistem Klasifikasi Ulasan",
    page_icon="📊",
    layout="wide"
)

# Custom CSS untuk UI sesuai revisi
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #9e9e9e !important;
        color: white !important;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        width: 200px;
    }

    .stButton > button:hover {
        background-color: #6e6e6e !important;
        color: white !important;
    }

    textarea {
        border-radius: 8px !important;
        padding: 12px !important;
    }

    /* Make the radio options left-aligned and bolder */
    ection[data-testid="stSidebar"] .stRadio > div {
    padding-left: 6px;
}
    section[data-testid="stSidebar"] .stRadio button[aria-pressed="true"] {
    font-weight: 700;
    background: rgba(255,255,255,0.06);
}
    </style>
    """,
    unsafe_allow_html=True
)

# ========== CLEANING TEXT FUNCTION ==========
def clean_text_ml(s: str) -> str:
    """Cleaning untuk NB & LightGBM"""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|\bRT\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_light(s: str) -> str:
    """Cleaning GRU (lebih ringan)"""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ========== LOAD ARTIFACT MODEL ==========
@st.cache_resource(show_spinner=True)
def load_artifacts():
    mdir = Path("models")

    tfidf = joblib.load(mdir / "tfidf.pkl")
    svd = joblib.load(mdir / "svd.pkl")
    nb = joblib.load(mdir / "nb.pkl")
    lgbm = joblib.load(mdir / "lgbm.pkl")

    # GRU
    with open(mdir / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    meta = joblib.load(mdir / "gru_meta.pkl")
    MAXLEN = int(meta.get("MAXLEN", 150))
    BEST_THR = float(meta.get("best_thr", 0.5))

    sm = tf.saved_model.load(str(mdir / "gru_savedmodel"))
    sig = sm.signatures.get("serving_default") or list(sm.signatures.values())[0]

    return tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, sig


# ========== PREDIKSI MODEL ==========
def predict_nb_lgbm(text, tfidf, svd, nb, lgbm):
    clean = clean_text_ml(text)
    X = tfidf.transform([clean])
    X_svd = svd.transform(X)
    return (int(nb.predict(X)[0]), nb.predict_proba(X)[0][1]), \
           (int(lgbm.predict(X_svd)[0]), lgbm.predict_proba(X_svd)[0][1])

def predict_gru(text, tok, MAXLEN, BEST_THR, gru_sig):
    """
    Robust caller for GRU SavedModel signature.
    Tries:
      1) call with int32 tensor
      2) call with float32 tensor
      3) call by wrapping into dictionary argument (if signature expects named input)
    Returns: (label:int, proba:float)
    """
    clean = clean_light(text)
    seq = tok.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    # convert to numpy array explicitly
    import numpy as _np
    pad_np = _np.asarray(pad)

    # helper to extract float probability from signature output
    def _extract_proba(output):
        # output may be a dict of tensors or a single tensor
        if isinstance(output, dict):
            val = list(output.values())[0]
        else:
            val = output
        # ensure it's a numpy scalar/array and return float
        try:
            arr = _np.asarray(val)
            # flatten and take first element
            return float(arr.reshape(-1)[0])
        except Exception:
            # fallback: try tensorflow conversion
            import tensorflow as _tf
            return float(_tf.reshape(val, [-1])[0].numpy())

    # 1) try int32 tensor positional
    import tensorflow as _tf
    try:
        inp = _tf.convert_to_tensor(pad_np, dtype=_tf.int32)
        out = gru_sig(inp)
        proba = _extract_proba(out)
        label = 1 if proba >= BEST_THR else 0
        return label, proba
    except Exception as e_int:
        # 2) try float32 positional
        try:
            inp = _tf.convert_to_tensor(pad_np, dtype=_tf.float32)
            out = gru_sig(inp)
            proba = _extract_proba(out)
            label = 1 if proba >= BEST_THR else 0
            return label, proba
        except Exception as e_float:
            # 3) try calling with named argument(s) from signature
            try:
                sig_inputs = gru_sig.structured_input_signature
                # structured_input_signature example: ((), {'input_1': TensorSpec(shape=(None, MAXLEN), dtype=tf.int32, name='input_1')})
                if isinstance(sig_inputs, tuple) and len(sig_inputs) >= 2 and isinstance(sig_inputs[1], dict):
                    name = list(sig_inputs[1].keys())[0]
                    # try int32 first
                    try:
                        out = gru_sig(**{name: _tf.convert_to_tensor(pad_np, dtype=_tf.int32)})
                        proba = _extract_proba(out)
                        label = 1 if proba >= BEST_THR else 0
                        return label, proba
                    except Exception:
                        out = gru_sig(**{name: _tf.convert_to_tensor(pad_np, dtype=_tf.float32)})
                        proba = _extract_proba(out)
                        label = 1 if proba >= BEST_THR else 0
                        return label, proba
                # if signature format unknown, re-raise
                raise RuntimeError("Cannot determine signature input names/types")
            except Exception as e_named:
                # If all attempts failed, raise a clear error including previous exceptions
                raise RuntimeError(
                    f"GRU inference failed. Tried int32 positional (err: {e_int}), "
                    f"float32 positional (err: {e_float}), and named-arg attempt (err: {e_named})."
                )

# ========== KOMPONEN OUTPUT BAR ==========
def render_row(model_name, proba, label):
    col1, col2, col3 = st.columns([2, 6, 2])
    with col1:
        st.write(f"**{model_name}**")
    with col2:
        st.progress(min(max(proba, 0), 1))
        st.caption(f"{int(round(proba * 100))}%")
    with col3:
        text = "Positif" if label == 1 else "Negatif"
        color = "#22c55e" if label == 1 else "#ef4444"
        st.markdown(
            f"<div style='padding:8px;border-radius:8px;text-align:center;color:white;background:{color}'>{text}</div>",
            unsafe_allow_html=True
        )

# Cek apakah artifacts berhasil dimuat
try:
    tfidf, svd, nb, lgbm, tok, MAXLEN, BEST_THR, GRU_SIG = load_artifacts()
    ARTIFACTS_OK = True
    LOAD_ERROR = ""
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)

# =========================================================
# Sidebar
# =========================================================
menu = st.sidebar.radio(
    label="",
    options=["Beranda", "Analisis", "Tentang"],
    index=0,
    label_visibility="collapsed"
)

# ✅ Tambahkan START di sini (SETELAH radio navigation)
st.sidebar.markdown("### Contoh File Dataset ")
sample_bytes = load_sample_csv()
if sample_bytes:
    st.sidebar.download_button(
        label="Download CSV",
        data=sample_bytes,
        file_name="contoh_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.sidebar.error("❌ File contoh dataset tidak ditemukan.")
st.sidebar.caption("File ini dapat diunduh dan digunakan langsung pada halaman 'Analisis' untuk demo.")

# =========================================================
# Halaman Beranda 
# =========================================================
if menu == "Beranda":
    st.markdown(
        "<h1 style='font-weight:700; text-align:center;'>Sistem Klasifikasi Ulasan</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='text-align:center; font-size:17px; padding-bottom: 10px;'>
        Sistem ini dirancang untuk menganalisis sentimen dari ulasan pengguna Tokopedia secara
        otomatis dan melakukan perbandingan hasil prediksi dari ketiga metode yang digunakan.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")

    # Tombol download Buku Manual
    st.markdown("##### Panduan Penggunaan:")

    st.markdown(
        """
        <p style='font-size:17px; padding-bottom: 10px;'>
        Panduan ini berisi instruksi lengkap untuk membantu Anda memanfaatkan semua fitur program secara maksimal
        </p>
        """,
        unsafe_allow_html=True,
    )
    pdf_path = Path("assets/Manual_Book.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Unduh Buku Panduan (PDF)",
                data=f,
                file_name="Manual_Book.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.warning("File manual belum ditemukan pada folder assets/Manual_Book.pdf.")

    st.write("---")

    # Input DEMO untuk Beranda (PASTIKAN INI ADA DI SINI, BUKAN DI LUAR)
    st.markdown("#### Silahkan coba untuk memasukkan sebuah ulasan:")

    text_demo = st.text_area(
        "",
        placeholder="Tulis ulasan di sini...",
        height=140,
    )

    # Tambahkan JARAK sebelum tombol Analisis
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    if st.button("Klasifikasi"):
        if text_demo.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            (nb_label, nb_proba), (lgb_label, lgb_proba) = predict_nb_lgbm(
                text_demo, tfidf, svd, nb, lgbm
            )
            gru_label, gru_proba = predict_gru(
                text_demo, tok, MAXLEN, BEST_THR, GRU_SIG
            )

            render_row("Naive Bayes", nb_proba, nb_label)
            render_row("LightGBM", lgb_proba, lgb_label)
            render_row("GRU", gru_proba, gru_label)

# =========================================================
# PAGE: ANALISIS 
# =========================================================
elif menu == "Analisis":
    st.markdown(
        "<h2 style='font-weight:700; text-align:center;'>Klasifikasi Ulasan dengan File</h2>",
        unsafe_allow_html=True,
    )

    # teks intro
    st.markdown(
        """
        <p style="font-size:17px; margin-bottom:10px;">
        Silahkan upload file dataset Anda dengan ketentuan sebagai berikut:
        </p>
        """,
        unsafe_allow_html=True
    )

    # ---- Syarat & Ketentuan (expandable) ----
    with st.expander("Syarat & Ketentuan (klik untuk buka)"):
        st.markdown(
            """
            **Syarat:**
            1. File dalam format CSV / XLSX (CSV direkomendasikan).
            2. Didalamnya hanya ada 2 Kolom yaitu Content dan Label.
            3. Kolom 'Label' harus berisi angka: 1 (untuk Positif) dan 0 (untuk Negatif).
            4. Pastikan tidak ada baris yang datanya kosong (tidak ada missing value).

            **Ketentuan:**
            1. Penamaan kolom harus sama persis (perhatikan huruf besar/kecil).
            2. Untuk performa terbaik, disarankan mengunggah data di bawah 1.000 baris. 
            (Ini agar proses analisisnya cepat dan tidak time out).
            """
        )

    # ---- Upload file (CSV / XLSX) ----
    uploaded = st.file_uploader("**Unggah file CSV**", type=["csv","xlsx"])
    st.caption("Jika ingin mencoba, bisa menggunakan dataset yang tersedia.")

    # Pilihan metode visualisasi / fokus
    method = st.selectbox("Pilih Metode (untuk visual/top-token)", ["Semua", "Naive Bayes", "LightGBM", "GRU"])

    if uploaded is None:
        st.info("Silakan unggah file untuk memulai analisis.")
    else:
        # baca file & normalisasi kolom (tolerant)
        try:
            if str(uploaded.name).lower().endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Gagal membaca file: " + str(e))
            st.stop()

        # tampilkan kolom (help)
        cols_list = list(df.columns)
        st.caption("Kolom pada file: " + ", ".join(cols_list))

        # mapping case-insensitive
        cols_lower = {c.lower(): c for c in df.columns}
        text_candidates = ["text", "content", "review", "comment", "ulasan", "isi"]
        score_candidates = ["score", "label", "rating", "nilai"]

        text_col = None
        for cand in text_candidates:
            if cand in cols_lower:
                text_col = cols_lower[cand]
                break

        if text_col is None:
            st.error("Tidak menemukan kolom teks. Pastikan file memiliki kolom seperti: text, content, review, comment, ulasan.")
            st.stop()

        score_col = None
        for cand in score_candidates:
            if cand in cols_lower:
                score_col = cols_lower[cand]
                break

        # buat kolom standar
        df = df.copy()
        df["text"] = df[text_col].astype(str).fillna("")
        if score_col is not None:
            df["score"] = df[score_col]

        st.success(f"File dimuat — {len(df)} baris (kolom teks: '{text_col}'{', kolom score: '+score_col if score_col else ''})")
        if len(df) > 20000:
            st.warning("File sangat besar (>20k baris). Proses dapat memakan waktu lama.")

        # tombol analisis
        if st.button("Analisis"):
            if not ARTIFACTS_OK:
                st.error("Model/artifact tidak tersedia. Error saat memuat: " + LOAD_ERROR)
                st.stop()

            N = len(df)
            pb = st.progress(0)
            results = []

            for i, txt in enumerate(df["text"].tolist()):
                # prediksi NB & LGBM
                try:
                    (nb_l, nb_p), (lgb_l, lgb_p) = predict_nb_lgbm(txt, tfidf, svd, nb, lgbm)
                except Exception:
                    nb_l, nb_p, lgb_l, lgb_p = 0, 0.0, 0, 0.0

                # prediksi GRU
                try:
                    gru_l, gru_p = predict_gru(txt, tok, MAXLEN, BEST_THR, GRU_SIG)
                except Exception:
                    gru_l, gru_p = 0, 0.0

                results.append({
                    "pred_nb": int(nb_l),
                    "prob_nb": float(nb_p),
                    "pred_lgbm": int(lgb_l),
                    "prob_lgbm": float(lgb_p),
                    "pred_gru": int(gru_l),
                    "prob_gru": float(gru_p),
                })

                if i % 20 == 0 or i == N-1:
                    pb.progress(min((i+1)/N, 1.0))

            # gabungkan hasil
            res_df = pd.DataFrame(results)
            df = pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)

            st.success("Analisis selesai — scroll ke bawah untuk melihat visualisasi dan tabel.")

            # -----------------------------
            # VISUALISASI: PIE CHART SENTIMEN
            # -----------------------------
            import matplotlib.pyplot as plt
            from collections import Counter

            st.subheader("Distribusi Sentimen Prediksi")

            def plot_pie_from_preds(preds, title):
                cnt = Counter(preds.astype(int))
                labels = []
                sizes = []
                if cnt.get(1, 0) > 0:
                    labels.append("Positif (1)")
                    sizes.append(cnt.get(1, 0))
                if cnt.get(0, 0) > 0:
                    labels.append("Negatif (0)")
                    sizes.append(cnt.get(0, 0))
                fig, ax = plt.subplots(figsize=(4,3))
                if len(sizes) == 0:
                    ax.text(0.5, 0.5, "Tidak ada data", ha="center")
                else:
                    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.set_title(title)
                st.pyplot(fig)

            # tentukan preds berdasarkan pilihan metode
            if method == "Naive Bayes":
                plot_pie_from_preds(df["pred_nb"].astype(int).values, "Naive Bayes")
            elif method == "LightGBM":
                plot_pie_from_preds(df["pred_lgbm"].astype(int).values, "LightGBM")
            elif method == "GRU":
                plot_pie_from_preds(df["pred_gru"].astype(int).values, "GRU")
            else:
                # majority / average across models
                avg_pred = df[["pred_nb","pred_lgbm","pred_gru"]].mean(axis=1).round().astype(int)
                plot_pie_from_preds(avg_pred.values, "Semua Model (majority/avg)")

            # -----------------------------
            # VISUALISASI: TOP 10 TOKEN
            # -----------------------------
            st.subheader("Top 10 Kata Teratas (Negatif | Positif)")

            from sklearn.feature_extraction.text import CountVectorizer
            from collections import Counter as _Counter

            def top_tokens_for_preds(texts_series, preds_array, topn=10, hide_oov=True):
                pos_texts = texts_series[preds_array == 1].astype(str).tolist()
                neg_texts = texts_series[preds_array == 0].astype(str).tolist()

                all_texts = pos_texts + neg_texts
                if len(all_texts) == 0:
                    return [], []

                vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words=None)
                vec.fit(all_texts)
                vocab = vec.get_feature_names_out()

                def counts_for_list(lst):
                    if len(lst) == 0:
                        return _Counter()
                    m = vec.transform(lst)
                    s = m.sum(axis=0)
                    return _Counter({vocab[idx]: int(s[0, idx]) for idx in range(len(vocab)) if s[0, idx] > 0})

                pos_ct = counts_for_list(pos_texts)
                neg_ct = counts_for_list(neg_texts)

                if hide_oov:
                    pos_ct.pop("<oov>", None)
                    neg_ct.pop("<oov>", None)

                pos_top = pos_ct.most_common(topn)
                neg_top = neg_ct.most_common(topn)
                return pos_top, neg_top

            # pilih preds sesuai method (fallback majority jika "Semua")
            if method == "Naive Bayes":
                preds_col = "pred_nb"
                preds_arr = df[preds_col].astype(int).values
            elif method == "LightGBM":
                preds_col = "pred_lgbm"
                preds_arr = df[preds_col].astype(int).values
            elif method == "GRU":
                preds_col = "pred_gru"
                preds_arr = df[preds_col].astype(int).values
            else:
                preds_col = "avg_majority"
                preds_arr = df[["pred_nb","pred_lgbm","pred_gru"]].mean(axis=1).round().astype(int).values

            hide_oov = st.session_state.get("hide_oov", True)
            pos_top, neg_top = top_tokens_for_preds(df["text"], preds_arr, topn=10, hide_oov=hide_oov)

            c1, c2 = st.columns(2)
            with c1:
                if len(neg_top) > 0:
                    neg_df_vis = pd.DataFrame(neg_top, columns=["token","count"]).set_index("token")
                    st.bar_chart(neg_df_vis)
                else:
                    st.write("Tidak ada token Negatif.")
            with c2:
                if len(pos_top) > 0:
                    pos_df_vis = pd.DataFrame(pos_top, columns=["token","count"]).set_index("token")
                    st.bar_chart(pos_df_vis)
                else:
                    st.write("Tidak ada token Positif.")

            st.write("---")

            # -----------------------------
            # TABEL EVALUASI: pakai hasil pelatihan (jika tersedia), sederhana
            # -----------------------------
            st.subheader("Tabel Perbandingan (hasil pelatihan / eksperimen)")

            exp_path = Path("models/experiments.json")
            fallback = [
                {"Model": "GRU", "Accuracy": 0.8725, "Precision": 0.8382, "Recall": 0.7267, "F1-Score": 0.7785},
                {"Model": "Naive Bayes", "Accuracy": 0.8632, "Precision": 0.8138, "Recall": 0.7213, "F1-Score": 0.7648},
                {"Model": "LightGBM", "Accuracy": 0.8422, "Precision": 0.7680, "Recall": 0.6992, "F1-Score": 0.7320},
            ]

            if exp_path.exists():
                try:
                    import json
                    exp = json.loads(exp_path.read_text(encoding="utf-8"))
                    def find(exp, name):
                        for k, v in exp.items():
                            if k.strip().lower() == name.lower():
                                return v
                            return None

                    rows = []
                    for model_name in ["GRU", "Naive Bayes", "LightGBM"]:
                        entry = find(exp, model_name)
                        if entry:
                            rows.append({
                                "Model": model_name,
                                "Accuracy": round(float(entry.get("accuracy", 0)), 4),
                                "Precision": round(float(entry.get("precision", 0)), 4),
                                "Recall": round(float(entry.get("recall", 0)), 4),
                                "F1-Score": round(float(entry.get("f1", entry.get("f1_score", 0))), 4),
                            })
                    if len(rows) == 0:
                        rows = fallback
                except Exception:
                    rows = fallback
            else:
                rows = fallback

            st.dataframe(pd.DataFrame(rows), use_container_width=True)


            # -----------------------------
            # TABEL EFFICIENCY (estimasi) + DOWNLOAD
            # -----------------------------
            st.subheader("Tabel Efisiensi Komputasi")
            eff_df = pd.DataFrame([
                {"Model":"GRU", "Waktu Training (s)": 876.44, "Waktu Inferensi (ms)": 13.59},
                {"Model":"Naive Bayes", "Waktu Training (s)": 0.010, "Waktu Inferensi (ms)": 0.001},
                {"Model":"LightGBM", "Waktu Training (s)": 39.91, "Waktu Inferensi (ms)": 0.095}
            ])
            st.dataframe(eff_df, use_container_width=True)

            # download hasil (UTF-8 BOM agar Excel kebaca)
            csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Unduh Hasil Analisis (CSV)", csv_bytes, file_name="hasil_analisis.csv", mime="text/csv")

# =========================================================
# PAGE 3 — TENTANG
# =========================================================
elif menu == "Tentang":
    st.markdown(
        "<h2 style='font-weight:700; text-align:center;'>Tentang Perancang</h2>",
        unsafe_allow_html=True,
    )
    st.write("---")
    col1, col2 = st.columns([2, 4])

    with col1:
        st.image("assets/foto.jpg", width=280)

    with col2:
        st.markdown("""
        **Nama:**  
        Aulia Dwi Yulianti  

        **NIM:**  
        5352021078  

        **Judul Skripsi:**  
        *Perancangan Sistem Klasifikasi Ulasan pada Aplikasi Tokopedia Menggunakan Metode Naive Bayes, LightGBM, dan GRU*  

        **Dosen Pembimbing:**  
        Tony, S.Kom., M.Kom., Ph.D.  

        **Abstrak:**  
        Ulasan pelanggan pada platform e-commerce seperti Tokopedia merupakan data krusial untuk wawasan bisnis, 
        namun volumenya yang besar menuntut adanya analisis sentimen otomatis. Perancangan ini bertujuan untuk 
        membangun sebuah prototipe sistem aplikasi web yang mampu melakukan klasifikasi sentimen pada ulasan 
        aplikasi Tokopedia secara otomatis. Sistem ini dirancang untuk mengimplementasikan dan membandingkan 
        kinerja dari tiga model yang mewakili paradigma machine learning berbeda yaitu metode probabilistik 
        Naïve Bayes, ensemble learning LightGBM, dan deep learning sekuensial Gated Recurrent Unit (GRU). 
        Data ulasan publik dikumpulkan dari Google Play Store melalui teknik web scraping dan melewati tahap 
        pra-pemrosesan teks sebelum diimplementasikan pada ketiga model. Kinerja setiap model dievaluasi menggunakan 
        akakurasi, presisi, recall, F1-score, serta efisiensi komputasi berupa waktu pelatihan dan inferensi.
        Hasil dari perancangan ini adalah prototipe fungsional yang menyajikan perbandingan antar model untuk menentukan
        metode yang paling optimal pada klasifikasi ulasan berbahasa Indonesia.
        """)

    # Penutup / footer kecil
    st.markdown(
        "<p style='text-align:center; color:gray; font-size:14px;'>© 2025 - Sistem Klasifikasi Ulasan </p>",
        unsafe_allow_html=True
    )