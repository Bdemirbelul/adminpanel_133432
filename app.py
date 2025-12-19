import streamlit as st
import io
import os
import time
import traceback
import builtins
import sys
from datetime import datetime
from contextlib import contextmanager

import pandas as pd

# Scraper'ları import et
from scrapers.company1 import run as run_company1
from scrapers.company2 import run as run_company2
from scrapers.company3 import run as run_company3
from scrapers.company4 import run as run_company4
from scrapers.company5 import run as run_company5
from scrapers.company6 import run as run_company6
from scrapers.company7 import run as run_company7

st.set_page_config(page_title="Yönetici Scraper Panel", layout="wide")

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_BASE, exist_ok=True)

COMPANIES = [
    ("Coldwell Banker", run_company1),
    ("Remax", run_company2),
    ("Century21", run_company3),
    ("ERA", run_company4),
    ("Dialog", run_company5),
    ("Turyap", run_company6),
    ("Rozky", run_company7),
]


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    st.session_state.logs.append(log_entry)


class TeeOutput:
    """Hem terminale hem de log'a yazan output wrapper"""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = ""
        self.log_container = None
        
    def set_log_container(self, container):
        """Log container'ı set et (gerçek zamanlı güncelleme için)"""
        self.log_container = container
        
    def write(self, text):
        if not text:
            return
            
        # Buffer'a ekle (çok satırlı çıktılar için)
        self.buffer += text
        
        # Eğer yeni satır karakteri varsa, satırları işle
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Son satır hariç hepsini log'a ekle (son satır henüz tamamlanmamış olabilir)
            for line in lines[:-1]:
                if line.strip():  # Boş satırları atla
                    log(line)
            # Kalan kısmı buffer'da tut
            self.buffer = lines[-1]
        # Eğer buffer çok uzunsa (yeni satır olmadan), zorla log'a ekle
        elif len(self.buffer) > 500:
            log(self.buffer.rstrip())
            self.buffer = ""
        
        # Orijinal stream'e yaz
        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except Exception:
            pass  # Stream kapatılmış olabilir
        
    def flush(self):
        # Buffer'da kalan varsa onu da log'a ekle
        if self.buffer.strip():
            log(self.buffer.strip())
            self.buffer = ""
        try:
            self.original_stream.flush()
        except Exception:
            pass
        
    def __getattr__(self, name):
        return getattr(self.original_stream, name)


@contextmanager
def capture_output(log_container=None):
    """Tüm stdout ve stderr çıktısını yakalayan context manager"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_print = builtins.print
    
    # Print fonksiyonunu override et
    def _print(*args, **kwargs):
        # Her zaman log'a yaz (file parametresine bakmadan)
        msg = " ".join(str(a) for a in args)
        if msg.strip():  # Boş mesajları atla
            log(msg)
        # Orijinal print'i çağır (file parametresini koruyarak)
        original_print(*args, **kwargs)
    
    try:
        # stdout ve stderr'i yakala
        tee_stdout = TeeOutput(original_stdout)
        tee_stderr = TeeOutput(original_stderr)
        if log_container is not None:
            tee_stdout.set_log_container(log_container)
            tee_stderr.set_log_container(log_container)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        builtins.print = _print
        yield
    finally:
        # Her durumda eski haline getir
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        builtins.print = original_print


def run_one(name, fn, out_dir, log_container=None):
    start = time.time()
    log(f"🚀 {name} çalışmaya başladı...")
    
    # Log container'ı güncelle
    if log_container is not None:
        log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
        log_container.code(log_text, language="", line_numbers=False)
    
    log(f"⏳ Lütfen bekleyin, veri çekiliyor...")
    if log_container is not None:
        log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
        log_container.code(log_text, language="", line_numbers=False)
    
    with capture_output(log_container):
        try:
            result = fn(out_dir)  # scraper'ın run() fonksiyonu
            elapsed = time.time() - start
            
            # İlerleme mesajları
            if elapsed > 10:
                log(f"⏳ Bitmeye yakın, veriler işleniyor...")
                if log_container is not None:
                    log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
                    log_container.code(log_text, language="", line_numbers=False)
            
            log(f"✅ Veriler alındı, dosyalanıyor...")
            if log_container is not None:
                log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
                log_container.code(log_text, language="", line_numbers=False)
            
            log(f"✓ {name} tamamlandı ({elapsed:.1f}s). {result}")
            
            # Log container'ı güncelle
            if log_container is not None:
                log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
                log_container.code(log_text, language="", line_numbers=False)
            
            return True
        except Exception:
            elapsed = time.time() - start
            log(f"❌ {name} hata oluştu ({elapsed:.1f}s):\n{traceback.format_exc()}")
            
            # Log container'ı güncelle
            if log_container is not None:
                log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs)
                log_container.code(log_text, language="", line_numbers=False)
            
            return False


if "logs" not in st.session_state:
    st.session_state.logs = []

# Tab başlıklarını büyük yapmak için CSS ve gri butonlar için CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 20px;
        font-weight: bold;
    }
    /* Remax, Dialog ve Turyap butonlarını gri yap ve isimlerini kırmızı yap */
    button[data-testid*="btn_Remax"],
    button[data-testid*="btn_Dialog"],
    button[data-testid*="btn_Turyap"] {
        background-color: #808080 !important;
        color: #ff0000 !important;
        opacity: 0.8;
    }
    button[data-testid*="btn_Remax"]:hover,
    button[data-testid*="btn_Dialog"]:hover,
    button[data-testid*="btn_Turyap"]:hover {
        background-color: #6a6a6a !important;
        color: #ff0000 !important;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

tab_scraper, tab_diff, tab_view = st.tabs(
    ["Scraper Paneli", "CSV/Excel Karşılaştırma", "Çıktıları Görüntüle"]
)

with tab_scraper:
    # Output klasörü (run bazlı)
    stamp = datetime.now().strftime("%Y-%m-%d")
    run_folder = stamp
    out_dir = os.path.join(OUTPUT_BASE, run_folder)
    os.makedirs(out_dir, exist_ok=True)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Sistemi Çalıştır")

        for name, fn in COMPANIES:
            if st.button(f"▶️ {name}", key=f"btn_{name}"):
                # Log'ları sıfırla
                st.session_state.logs = []
                
                # Sağ taraftaki log container'ı kullan
                with right:
                    st.subheader("Log")
                    log_container = st.empty()
                    log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs) if st.session_state.logs else ""
                    log_container.code(log_text, language="", line_numbers=False)
                
                run_one(name, fn, out_dir, log_container)

    with right:
        st.subheader("Log")
        # Log container'ı oluştur
        log_container = st.empty()
        
        # Log'ları göster
        log_text = "\n".join(st.session_state.logs[-500:]) if len(st.session_state.logs) > 500 else "\n".join(st.session_state.logs) if st.session_state.logs else ""
        log_container.code(log_text, language="", line_numbers=False)

    st.divider()

    # Çıktı dosyalarını listeler
    st.subheader("Üretilen Dosyalar")

    files = []
    for root, _, filenames in os.walk(out_dir):
        for f in filenames:
            files.append(os.path.join(root, f))

    if not files:
        st.info("Henüz dosya yok.")
    else:
        for fpath in sorted(files):
            st.write("•", os.path.relpath(fpath, OUTPUT_BASE))

with tab_diff:
    # CSS for larger file uploader labels
    st.markdown("""
    <style>
        div[data-testid="stFileUploader"] label {
            font-size: 20px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)

    file_a = st.file_uploader(
        "Dosya A yükle (.csv / .xlsx)", type=["csv", "xlsx"], key="a"
    )
    file_b = st.file_uploader(
        "Dosya B yükle (.csv / .xlsx)", type=["csv", "xlsx"], key="b"
    )

    mode = st.radio(
        "Karşılaştırma modu",
        ["Tek kolon (değer listesi)", "Tam satır (row diff)"],
        horizontal=True,
    )

    ignore_case = st.checkbox("Büyük/küçük harf duyarsız", value=True)
    trim = st.checkbox("Boşlukları kırp (strip)", value=True)

    def read_csv_smart(uploaded) -> pd.DataFrame:
        raw = uploaded.getvalue()
        txt = raw.decode("utf-8", errors="replace")
        for sep in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(
                    io.StringIO(txt), sep=sep, dtype=str, engine="python"
                )
                if df.shape[1] >= 2 or (df.shape[1] == 1 and len(df) > 0):
                    return df.dropna(how="all")
            except Exception:
                continue
        return (
            pd.read_csv(io.StringIO(txt), dtype=str, engine="python").dropna(
                how="all"
            )
        )

    def read_excel(uploaded) -> pd.DataFrame:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox(
            f"Sheet seç ({uploaded.name})", xls.sheet_names, key=uploaded.name
        )
        df = pd.read_excel(uploaded, sheet_name=sheet, dtype=str)
        return df.dropna(how="all")

    def read_any(uploaded) -> pd.DataFrame:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            return read_csv_smart(uploaded)
        if name.endswith(".xlsx"):
            return read_excel(uploaded)
        raise ValueError("Desteklenmeyen dosya tipi")

    def norm_series(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str)
        if trim:
            s = s.str.strip()
        if ignore_case:
            s = s.str.lower()
        return s

    def norm_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = norm_series(out[c])
        return out

    if file_a and file_b:
        col1, col2 = st.columns(2)
        with col1:
            df_a = read_any(file_a)
            st.caption(
                f"A: {file_a.name} | satır: {len(df_a)} | kolon: {len(df_a.columns)}"
            )
            st.dataframe(df_a.head(20), use_container_width=True)
        with col2:
            df_b = read_any(file_b)
            st.caption(
                f"B: {file_b.name} | satır: {len(df_b)} | kolon: {len(df_b.columns)}"
            )
            st.dataframe(df_b.head(20), use_container_width=True)

        st.divider()

        if mode == "Tek kolon (değer listesi)":
            common_cols = sorted(set(df_a.columns) & set(df_b.columns))
            if not common_cols:
                st.error(
                    "İki dosyada ortak kolon yok. Kolon isimlerini eşitle veya 'Tam satır' modunu kullan."
                )
                st.stop()

            key_col = st.selectbox("Karşılaştırılacak kolon", common_cols)

            a_vals = set(norm_series(df_a[key_col]))
            b_vals = set(norm_series(df_b[key_col]))

            only_a = sorted([x for x in (a_vals - b_vals) if x != ""])
            only_b = sorted([x for x in (b_vals - a_vals) if x != ""])

            c1, c2 = st.columns(2)
            c1.metric("A’da var, B’de yok", len(only_a))
            c2.metric("B’de var, A’da yok", len(only_b))

            left, right = st.columns(2)
            with left:
                st.subheader(" A’da var, B’de yok")
                out_a = pd.DataFrame({key_col: only_a})
                st.dataframe(out_a, use_container_width=True)
                st.download_button(
                    "CSV indir (A-B)",
                    out_a.to_csv(index=False).encode("utf-8-sig"),
                    file_name="only_in_A.csv",
                    mime="text/csv",
                )

            with right:
                st.subheader(" B’de var, A’da yok")
                out_b = pd.DataFrame({key_col: only_b})
                st.dataframe(out_b, use_container_width=True)
                st.download_button(
                    "CSV indir (B-A)",
                    out_b.to_csv(index=False).encode("utf-8-sig"),
                    file_name="only_in_B.csv",
                    mime="text/csv",
                )

        else:
            cols_a = st.multiselect(
                "A: Hangi kolonlar dahil olsun? (boşsa hepsi)",
                df_a.columns.tolist(),
                default=[],
            )
            cols_b = st.multiselect(
                "B: Hangi kolonlar dahil olsun? (boşsa hepsi)",
                df_b.columns.tolist(),
                default=[],
            )

            use_a = df_a[cols_a] if cols_a else df_a
            use_b = df_b[cols_b] if cols_b else df_b

            na = norm_df(use_a)
            nb = norm_df(use_b)

            rows_a = set(map(tuple, na.fillna("").to_numpy()))
            rows_b = set(map(tuple, nb.fillna("").to_numpy()))

            only_a = rows_a - rows_b
            only_b = rows_b - rows_a

            st.metric("A’da var, B’de yok (satır)", len(only_a))
            st.metric("B’de var, A’da yok (satır)", len(only_b))

            out_only_a = pd.DataFrame(list(only_a), columns=na.columns)
            out_only_b = pd.DataFrame(list(only_b), columns=nb.columns)

            l, r = st.columns(2)
            with l:
                st.subheader(" A’da var, B’de yok (satırlar)")
                st.dataframe(out_only_a, use_container_width=True)
                st.download_button(
                    "CSV indir (A-B rows)",
                    out_only_a.to_csv(index=False).encode("utf-8-sig"),
                    file_name="rows_only_in_A.csv",
                    mime="text/csv",
                )
            with r:
                st.subheader(" B’de var, A’da yok (satırlar)")
                st.dataframe(out_only_b, use_container_width=True)
                st.download_button(
                    "CSV indir (B-A rows)",
                    out_only_b.to_csv(index=False).encode("utf-8-sig"),
                    file_name="rows_only_in_B.csv",
                    mime="text/csv",
                )
    else:
        st.info("Başlamak için iki dosyayı yükle (CSV veya Excel).")


with tab_view:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>Çıktı Dosyalarını Görüntüle</h1>", unsafe_allow_html=True)

    # OUTPUT_BASE altındaki run klasörlerini listele
    run_dirs = [
        d
        for d in os.listdir(OUTPUT_BASE)
        if os.path.isdir(os.path.join(OUTPUT_BASE, d))
    ]

    if not run_dirs:
        st.info("Henüz oluşturulmuş çıktı klasörü yok.")
    else:
        run_dirs = sorted(run_dirs, reverse=True)
        selected_run = st.selectbox("Run klasörü seç", run_dirs)

        run_path = os.path.join(OUTPUT_BASE, selected_run)

        # Seçilen run klasörü altındaki CSV/Excel dosyalarını bul
        data_files = []
        for root, _, filenames in os.walk(run_path):
            for f in filenames:
                lower = f.lower()
                if lower.endswith(".csv") or lower.endswith(".xlsx"):
                    full_path = os.path.join(root, f)
                    rel = os.path.relpath(full_path, run_path)
                    data_files.append((rel, full_path))

        if not data_files:
            st.warning("Bu run klasöründe CSV veya Excel dosyası bulunamadı.")
        else:
            labels = [x[0] for x in data_files]
            choice = st.selectbox("Dosya seç", labels)

            chosen_path = dict(data_files)[choice]

            st.caption(f"Seçilen dosya: `{chosen_path}`")

            # Dosyayı oku
            try:
                if chosen_path.lower().endswith(".csv"):
                    # Farklı encoding'leri dene (Türkçe karakter desteği için)
                    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1254", "iso-8859-9"]
                    txt = None
                    used_encoding = None
                    
                    for enc in encodings:
                        try:
                            with open(chosen_path, "r", encoding=enc, errors="replace") as f:
                                txt = f.read()
                            used_encoding = enc
                            break
                        except Exception:
                            continue
                    
                    if txt is None:
                        st.error("Dosya okunamadı. Encoding sorunu olabilir.")
                        st.stop()
                    
                    # Ayracı otomatik anlamaya çalış
                    df_view = None
                    for sep in [";", ",", "\t", "|"]:
                        try:
                            test_df = pd.read_csv(
                                io.StringIO(txt), 
                                sep=sep, 
                                dtype=str, 
                                engine="python",
                                quotechar='"',
                                skipinitialspace=True
                            )
                            # Eğer birden fazla kolon varsa veya tek kolon ama çok satır varsa geçerli
                            if test_df.shape[1] >= 2 or (test_df.shape[1] == 1 and len(test_df) > 0):
                                df_view = test_df
                                break
                        except Exception:
                            continue
                    
                    if df_view is None:
                        # Son çare: pandas'ın otomatik algılamasına bırak
                        df_view = pd.read_csv(
                            io.StringIO(txt), 
                            dtype=str, 
                            engine="python",
                            quotechar='"'
                        )
                else:
                    df_view = pd.read_excel(chosen_path, dtype=str)

                df_view = df_view.dropna(how="all")

                # Eğer tüm satır tek bir kolonda "1,Ad,Mail,Telefon,Url" gibi duruyorsa,
                # bu tek kolonu virgülden bölerek ayrı kolonlara aç
                if df_view.shape[1] == 1:
                    first_col = df_view.columns[0]
                    sample_vals = df_view[first_col].dropna().astype(str).head(10)
                    # Satırların çoğunda virgül varsa split etmeyi dene
                    if any("," in v for v in sample_vals):
                        expanded = df_view[first_col].astype(str).str.split(
                            ",", expand=True
                        )
                        # Kolon isimlerini basit şekilde ver
                        expanded.columns = [
                            f"kolon_{i+1}" for i in range(expanded.shape[1])
                        ]
                        df_view = expanded

                st.caption(
                    f"Satır: {len(df_view)} | Kolon: {len(df_view.columns)}"
                )
                if chosen_path.lower().endswith(".csv") and used_encoding:
                    st.caption(f"Encoding: {used_encoding}")

                # Kolon seçimi
                all_cols = df_view.columns.tolist()
                selected_cols = st.multiselect(
                    "Gösterilecek kolonlar",
                    all_cols,
                    default=all_cols,
                )

                if selected_cols:
                    # Geniş kolonlar için column_config ile text wrap ekle
                    column_config = {}
                    for col in selected_cols:
                        column_config[col] = st.column_config.TextColumn(
                            col,
                            width="medium",
                            help=f"{col} kolonu"
                        )
                    
                    st.dataframe(
                        df_view[selected_cols],
                        use_container_width=True,
                        column_config=column_config,
                        hide_index=False,
                    )
                else:
                    st.info("En az bir kolon seçmelisin.")
            except Exception as e:
                st.error(f"Dosya okunurken hata oluştu: {e}")
