"""
Construction PPE Detection System
AI-Based Personal Protective Equipment Monitoring
Powered by YOLOv8 and Streamlit
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import io
import time
from datetime import datetime


# ============================================================
# 1. CUSTOM CSS STYLES
# ============================================================
def inject_custom_css():
    """Inject custom CSS for modern, professional industrial dashboard UI."""
    css = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        /* ---- Global Font ---- */
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* ---- Hide Default Elements ---- */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ---- App Background ---- */
        .stApp {
            background-color: #F5F7FA !important;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: #334155 !important;
        }

        /* ---- Card Container ---- */
        .card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #F1F5F9;
            margin-bottom: 20px;
            transition: box-shadow 0.2s ease;
        }
        .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        }

        /* ---- Card Title ---- */
        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-title .icon {
            font-size: 20px;
        }

        /* ---- Metric Card ---- */
        .metric-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 20px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            border: 1px solid #F1F5F9;
            text-align: center;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #2563EB;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }

        /* ---- Status Badge ---- */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 24px;
            border-radius: 100px;
            font-weight: 600;
            font-size: 15px;
        }
        .status-safe {
            background: #F0FDF4;
            color: #16A34A;
            border: 1px solid #BBF7D0;
        }
        .status-partial {
            background: #FEFCE8;
            color: #CA8A04;
            border: 1px solid #FDE68A;
        }
        .status-unsafe {
            background: #FEF2F2;
            color: #DC2626;
            border: 1px solid #FECACA;
        }

        /* ---- Buttons ---- */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
            box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            color: #475569 !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }
        .stButton > button[kind="secondary"]:hover {
            background: #F8FAFC !important;
            border-color: #CBD5E1 !important;
        }

        /* ---- File Uploader ---- */
        .stFileUploader > div > div {
            background: #F8FAFC !important;
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            transition: all 0.2s ease;
        }
        .stFileUploader > div > div:hover {
            border-color: #2563EB !important;
            background: #EAF4FF !important;
        }

        /* ---- Selectbox ---- */
        .stSelectbox > div > div {
            border-radius: 10px !important;
            border-color: #E2E8F0 !important;
        }

        /* ---- Progress Bar ---- */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2563EB, #22C55E) !important;
            border-radius: 100px !important;
        }

        /* ---- Compliance Bar ---- */
        .compliance-bar-container {
            background: #F1F5F9;
            border-radius: 100px;
            height: 12px;
            overflow: hidden;
            margin-top: 16px;
        }
        .compliance-bar-fill {
            height: 100%;
            border-radius: 100px;
            transition: width 0.6s ease;
        }

        /* ---- Sidebar Helpers ---- */
        .sidebar-divider {
            height: 1px;
            background: #E2E8F0;
            margin: 20px 0;
        }
        .sidebar-label {
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        /* ---- History Item ---- */
        .history-item {
            background: #F8FAFC;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            border: 1px solid #F1F5F9;
            transition: background 0.15s ease;
        }
        .history-item:hover {
            background: #F1F5F9;
        }
        .history-time {
            font-size: 11px;
            color: #94A3B8;
        }
        .history-filename {
            font-size: 13px;
            font-weight: 500;
            color: #334155;
            margin: 2px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .history-status {
            font-size: 12px;
            font-weight: 600;
        }

        /* ---- Header ---- */
        .app-header {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border-bottom: 1px solid #E2E8F0;
            padding: 20px 32px;
            margin: -16px -16px 24px -16px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .app-header .logo {
            font-size: 40px;
            line-height: 1;
        }
        .app-header h1 {
            font-size: 22px;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
            line-height: 1.2;
        }
        .app-header p {
            font-size: 13px;
            color: #64748B;
            margin: 4px 0 0 0;
            font-weight: 400;
        }

        /* ---- Summary Grid ---- */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .summary-card {
            background: linear-gradient(135deg, #EAF4FF 0%, #FFFFFF 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #DBEAFE;
            transition: transform 0.15s ease;
        }
        .summary-card:hover {
            transform: translateY(-2px);
        }
        .summary-card .value {
            font-size: 26px;
            font-weight: 700;
            color: #2563EB;
        }
        .summary-card .label {
            font-size: 12px;
            color: #64748B;
            font-weight: 500;
            margin-top: 4px;
        }

        /* ---- Metrics Grid ---- */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        /* ---- Stats Inner Grid ---- */
        .stats-inner-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            text-align: center;
        }
        .stats-inner-grid .stat-val {
            font-size: 28px;
            font-weight: 700;
            line-height: 1.2;
        }
        .stats-inner-grid .stat-lbl {
            font-size: 12px;
            color: #94A3B8;
            font-weight: 500;
            margin-top: 2px;
        }

        /* ---- Notification ---- */
        .notification {
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }
        .notification-success {
            background: #F0FDF4;
            color: #16A34A;
            border: 1px solid #BBF7D0;
        }
        .notification-warning {
            background: #FEFCE8;
            color: #CA8A04;
            border: 1px solid #FDE68A;
        }
        .notification-error {
            background: #FEF2F2;
            color: #DC2626;
            border: 1px solid #FECACA;
        }

        /* ---- Welcome State ---- */
        .welcome-state {
            text-align: center;
            padding: 80px 40px;
        }
        .welcome-state .icon-large {
            font-size: 72px;
            margin-bottom: 20px;
            display: block;
        }
        .welcome-state h2 {
            font-size: 24px;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 8px;
        }
        .welcome-state p {
            font-size: 15px;
            color: #64748B;
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.7;
        }

        /* ---- Footer ---- */
        .app-footer {
            text-align: center;
            padding: 24px;
            color: #94A3B8;
            font-size: 13px;
            border-top: 1px solid #E2E8F0;
            margin-top: 40px;
        }

        /* ---- Download Buttons Override ---- */
        .stDownloadButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            border: 1px solid #E2E8F0 !important;
            transition: all 0.2s ease !important;
        }
        .stDownloadButton > button:hover {
            border-color: #2563EB !important;
            color: #2563EB !important;
        }

        /* ---- Responsive Tweaks ---- */
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: repeat(3, 1fr) !important;
            }
            .summary-grid {
                grid-template-columns: 1fr !important;
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Construction PPE Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS sebelum elemen lain dirender
inject_custom_css()


# ============================================================
# 3. SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Inisialisasi semua variabel session state."""
    defaults = {
        "history": [],
        "total_processed": 0,
        "total_violations": 0,
        "compliance_sum": 0.0,
        "compliance_count": 0,
        "analyzed": False,
        "current_counts": {},
        "current_compliance": 0.0,
        "current_status": "",
        "current_color": "",
        "current_violations": 0,
        "current_table": None,
        "current_annotated": None,
        "current_image": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# 4. LOAD YOLO MODEL (Cached)
# ============================================================
@st.cache_resource
def load_yolo_model():
    """Memuat model YOLOv8 dengan caching agar tidak reload setiap interaksi."""
    try:
        model = YOLO("weights/best.pt")
        return model
    except Exception as e:
        return None


model = load_yolo_model()

if model is None:
    st.markdown("""
    <div class="notification notification-error" style="margin:24px 16px;">
        ❌ Gagal memuat model YOLO. Pastikan file <strong>weights/best.pt</strong> tersedia.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# 5. HELPER / UTILITY FUNCTIONS
# ============================================================

# Daftar kelas APD yang wajib dimiliki setiap pekerja
PPE_REQUIRED = ["Helmet", "Vest", "Gloves", "Boots"]


def calculate_compliance(counts: dict, person_count: int) -> tuple:
    """
    Menghitung persentase kepatuhan APD.

    Returns:
        tuple: (percentage, status_text, hex_color)
    """
    if person_count == 0:
        return 100.0, "Aman", "#22C55E"

    required = person_count * len(PPE_REQUIRED)
    found = sum(counts.get(cls, 0) for cls in PPE_REQUIRED)
    pct = min(100.0, (found / required) * 100) if required > 0 else 100.0

    if pct >= 90:
        return pct, "Aman", "#22C55E"
    elif pct >= 50:
        return pct, "Sebagian", "#EAB308"
    else:
        return pct, "Tidak Aman", "#EF4444"


def calculate_violations(counts: dict, person_count: int) -> int:
    """
    Menghitung total pelanggaran APD.
    Setiap APD yang kurang dari jumlah pekerja dihitung sebagai pelanggaran.
    """
    if person_count == 0:
        return 0
    violations = 0
    for ppe in PPE_REQUIRED:
        missing = max(0, person_count - counts.get(ppe, 0))
        violations += missing
    return violations


def build_detection_table(result, names: dict) -> pd.DataFrame:
    """Membuat DataFrame dari hasil deteksi YOLO."""
    rows = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        rows.append({
            "No": i + 1,
            "Class": names[cls_id],
            "Confidence": f"{conf:.2%}",
            "Bounding Box": f"[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]"
        })
    return pd.DataFrame(rows)


def get_status_css_class(status: str) -> str:
    """Mengembalikan kelas CSS berdasarkan status kepatuhan."""
    return {
        "Aman": "status-safe",
        "Sebagian": "status-partial",
        "Tidak Aman": "status-unsafe"
    }.get(status, "status-safe")


def get_status_emoji(status: str) -> str:
    """Mengembalikan emoji berdasarkan status kepatuhan."""
    return {
        "Aman": "✅",
        "Sebagian": "⚠️",
        "Tidak Aman": "❌"
    }.get(status, "✅")


def add_to_history(filename: str, counts: dict, compliance: float,
                   status: str, color: str, violations: int):
    """Menambahkan hasil analisis ke riwayat prediksi (maks 10 entri)."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename": filename,
        "counts": counts.copy(),
        "compliance": compliance,
        "status": status,
        "color": color,
        "violations": violations,
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]


def reset_analysis():
    """Mereset state analisis gambar saat ini."""
    keys_to_reset = [
        "analyzed", "current_counts", "current_compliance",
        "current_status", "current_color", "current_violations",
        "current_table", "current_annotated", "current_image",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            if key == "current_counts" or key == "current_table":
                st.session_state[key] = {} if key == "current_counts" else None
            elif key in ("current_compliance", "current_violations"):
                st.session_state[key] = 0.0 if key == "current_compliance" else 0
            elif key in ("current_status", "current_color"):
                st.session_state[key] = ""
            else:
                st.session_state[key] = None


# ============================================================
# 6. HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <span class="logo">🪖</span>
    <div>
        <h1>Construction PPE Detection System</h1>
        <p>AI-Based Personal Protective Equipment Monitoring</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. SIDEBAR
# ============================================================
with st.sidebar:

    # --- Upload Gambar ---
    st.markdown('<div class="sidebar-label">Upload Gambar</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Parameter Deteksi ---
    st.markdown('<div class="sidebar-label">Parameter Deteksi</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Skor confidence minimum untuk mendeteksi objek"
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Threshold Intersection over Union untuk Non-Maximum Suppression"
    )

    device_option = st.selectbox(
        "Device",
        options=["CPU", "GPU"],
        index=0,
        help="Pilih perangkat komputasi untuk inferensi"
    )
    device = "cpu" if device_option == "CPU" else "cuda:0"

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Tombol Aksi ---
    st.markdown('<div class="sidebar-label">Aksi</div>', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        analyze_btn = st.button(
            "🔍 Analisis",
            type="primary",
            use_container_width=True
        )
    with col_btn2:
        reset_btn = st.button(
            "↺ Reset",
            type="secondary",
            use_container_width=True
        )

    if reset_btn:
        reset_analysis()
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Informasi Model ---
    st.markdown('<div class="sidebar-label">Informasi Model</div>', unsafe_allow_html=True)
    class_list = ", ".join(model.names.values())
    st.markdown(f"""
    <div style="font-size:13px; color:#475569; line-height:1.8;">
        <div><strong>Nama Model:</strong> best.pt</div>
        <div><strong>Versi YOLO:</strong> v8</div>
        <div><strong>Jumlah Class:</strong> {len(model.names)}</div>
        <div style="margin-top:6px; color:#94A3B8; font-size:11px; line-height:1.5;">
            {class_list}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Riwayat Prediksi ---
    st.markdown('<div class="sidebar-label">Riwayat Prediksi</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div style="font-size:13px; color:#94A3B8; text-align:center; padding:20px 0;">'
            'Belum ada riwayat</div>',
            unsafe_allow_html=True
        )
    else:
        for entry in st.session_state.history:
            emoji = get_status_emoji(entry["status"])
            st.markdown(f"""
            <div class="history-item">
                <div class="history-time">{entry['timestamp']}</div>
                <div class="history-filename" title="{entry['filename']}">{entry['filename']}</div>
                <div class="history-status" style="color:{entry['color']}">
                    {emoji} {entry['status']} &middot; {entry['compliance']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# 8. DASHBOARD SUMMARY (selalu tampil)
# ============================================================
avg_compliance = (
    st.session_state.compliance_sum / st.session_state.compliance_count
    if st.session_state.compliance_count > 0
    else 0.0
)

st.markdown(f"""
<div class="summary-grid">
    <div class="summary-card">
        <div class="value">{st.session_state.total_processed}</div>
        <div class="label">Total Gambar Diproses</div>
    </div>
    <div class="summary-card">
        <div class="value">{st.session_state.total_violations}</div>
        <div class="label">Total Pelanggaran</div>
    </div>
    <div class="summary-card">
        <div class="value">{avg_compliance:.1f}%</div>
        <div class="label">Tingkat Kepatuhan Rata-rata</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 9. LOGIKA ANALISIS GAMBAR
# ============================================================
if analyze_btn and uploaded_file is None:
    st.markdown("""
    <div class="notification notification-warning">
        ⚠️ Silakan upload gambar terlebih dahulu sebelum melakukan analisis.
    </div>
    """, unsafe_allow_html=True)

elif analyze_btn and uploaded_file is not None:
    # Reset hasil sebelumnya
    reset_analysis()

    # Buka gambar
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.markdown(f"""
        <div class="notification notification-error">
            ❌ Gagal membaca gambar: {e}
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.session_state.current_image = image

    # Simpan ke file sementara untuk YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        tmp_path = tmp.name

    # Progress bar
    progress_bar = st.progress(0, text="Memulai analisis...")

    with st.spinner("Sedang melakukan deteksi APD..."):
        # Simulasi progress visual
        for p in [15, 35, 55, 75]:
            progress_bar.progress(p, text=f"Memproses gambar... {p}%")
            time.sleep(0.12)

        try:
            results = model.predict(
                source=tmp_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                save=False,
                verbose=False,
            )

            progress_bar.progress(90, text="Menyusun hasil deteksi...")
            time.sleep(0.1)

            result = results[0]
            names = model.names

            # Gambar dengan bounding box
            annotated = result.plot()
            st.session_state.current_annotated = annotated

            # Hitung jumlah per kelas
            counts = {}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                counts[label] = counts.get(label, 0) + 1

            st.session_state.current_counts = counts

            # Hitung kepatuhan & pelanggaran
            person_count = counts.get("Person", 0)
            compliance, status, color = calculate_compliance(counts, person_count)
            violations = calculate_violations(counts, person_count)

            st.session_state.current_compliance = compliance
            st.session_state.current_status = status
            st.session_state.current_color = color
            st.session_state.current_violations = violations

            # Buat tabel deteksi
            st.session_state.current_table = build_detection_table(result, names)

            # Tandai sudah dianalisis
            st.session_state.analyzed = True

            # Update statistik global
            st.session_state.total_processed += 1
            st.session_state.total_violations += violations
            st.session_state.compliance_sum += compliance
            st.session_state.compliance_count += 1

            # Simpan ke riwayat
            add_to_history(
                filename=uploaded_file.name,
                counts=counts,
                compliance=compliance,
                status=status,
                color=color,
                violations=violations,
            )

            progress_bar.progress(100, text="Analisis selesai!")
            time.sleep(0.25)
            progress_bar.empty()

        except Exception as e:
            progress_bar.empty()
            st.markdown(f"""
            <div class="notification notification-error">
                ❌ Error saat deteksi: {e}
            </div>
            """, unsafe_allow_html=True)
            st.session_state.analyzed = False


# ============================================================
# 10. TAMPILAN HASIL / WELCOME STATE
# ============================================================
if st.session_state.analyzed:

    counts = st.session_state.current_counts
    person_count = counts.get("Person", 0)
    total_objects = sum(counts.values())

    # ---- Baris Gambar ----
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("""
        <div class="card">
            <div class="card-title"><span class="icon">📷</span> Gambar Asli</div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.current_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_img2:
        st.markdown("""
        <div class="card">
            <div class="card-title"><span class="icon">🔍</span> Hasil Deteksi</div>
        """, unsafe_allow_html=True)
        st.image(st.session_state.current_annotated, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Baris Metrik APD ----
    ppe_display = [
        ("Helmet", "🪖"), ("Vest", "🦺"), ("Gloves", "🧤"),
        ("Boots", "👢"), ("Person", "👷"),
    ]

    metrics_html = '<div class="metrics-grid">'
    for cls_name, emoji in ppe_display:
        val = counts.get(cls_name, 0)
        metrics_html += f"""
        <div class="metric-card">
            <div style="font-size:26px; margin-bottom:6px;">{emoji}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{cls_name}</div>
        </div>
        """
    metrics_html += "</div>"
    st.markdown(metrics_html, unsafe_allow_html=True)

    # ---- Baris Kepatuhan & Statistik ----
    col_comp, col_stats = st.columns(2)

    with col_comp:
        status_class = get_status_css_class(st.session_state.current_status)
        status_emoji = get_status_emoji(st.session_state.current_status)
        compliance = st.session_state.current_compliance
        color = st.session_state.current_color

        st.markdown(f"""
        <div class="card">
            <div class="card-title"><span class="icon">🛡️</span> Status Kepatuhan APD</div>
            <div style="text-align:center; padding:20px 0;">
                <div class="status-badge {status_class}">
                    {status_emoji} {st.session_state.current_status}
                </div>
                <div style="font-size:52px; font-weight:800; color:{color}; margin:20px 0 4px 0;">
                    {compliance:.1f}%
                </div>
                <div style="font-size:13px; color:#94A3B8;">Tingkat Kepatuhan</div>
                <div class="compliance-bar-container">
                    <div class="compliance-bar-fill" style="width:{compliance}%; background:{color};"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        violations = st.session_state.current_violations
        st.markdown(f"""
        <div class="card">
            <div class="card-title"><span class="icon">📊</span> Statistik Deteksi</div>
            <div class="stats-inner-grid">
                <div>
                    <div class="stat-val" style="color:#2563EB;">{total_objects}</div>
                    <div class="stat-lbl">Total Objek</div>
                </div>
                <div>
                    <div class="stat-val" style="color:#2563EB;">{person_count}</div>
                    <div class="stat-lbl">Total Pekerja</div>
                </div>
                <div>
                    <div class="stat-val" style="color:#EF4444;">{violations}</div>
                    <div class="stat-lbl">Pelanggaran</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Tabel Hasil Deteksi ----
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="icon">📋</span> Tabel Hasil Deteksi</div>
    """, unsafe_allow_html=True)

    if st.session_state.current_table is not None and len(st.session_state.current_table) > 0:
        st.dataframe(
            st.session_state.current_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "No": st.column_config.NumberColumn(width="small"),
                "Class": st.column_config.TextColumn(width="medium"),
                "Confidence": st.column_config.TextColumn(width="medium"),
                "Bounding Box": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("Tidak ada objek yang terdeteksi pada gambar ini.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Download Hasil ----
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="icon">📥</span> Download Hasil</div>
    """, unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Download gambar hasil deteksi
        img_pil = Image.fromarray(st.session_state.current_annotated)
        img_buf = io.BytesIO()
        img_pil.save(img_buf, format="PNG")
        img_buf.seek(0)
        st.download_button(
            label="📥 Download Gambar (PNG)",
            data=img_buf,
            file_name="ppe_detection_result.png",
            mime="image/png",
            use_container_width=True,
        )

    with col_dl2:
        # Download CSV
        if st.session_state.current_table is not None and len(st.session_state.current_table) > 0:
            csv_buf = io.StringIO()
            st.session_state.current_table.to_csv(csv_buf, index=False)
            csv_data = csv_buf.getvalue()
        else:
            csv_data = "No,No,Class,Confidence,Bounding Box\n"

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="ppe_detection_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Notifikasi Sukses ----
    st.markdown("""
    <div class="notification notification-success">
        ✅ Analisis berhasil dilakukan. Hasil deteksi ditampilkan di atas.
    </div>
    """, unsafe_allow_html=True)

    # Jika ada pelanggaran, tampilkan warning tambahan
    if st.session_state.current_violations > 0:
        st.markdown(f"""
        <div class="notification notification-warning">
            ⚠️ Ditemukan <strong>{st.session_state.current_violations}</strong> pelanggaran APD. 
            Segera lakukan tindakan korektif.
        </div>
        """, unsafe_allow_html=True)

else:
    # ---- Welcome State ----
    st.markdown("""
    <div class="card">
        <div class="welcome-state">
            <span class="icon-large">🪖</span>
            <h2>Selamat Datang di PPE Detection System</h2>
            <p>
                Upload gambar proyek konstruksi melalui sidebar, atur parameter deteksi 
                (Confidence &amp; IoU Threshold), lalu klik tombol 
                <strong>"🔍 Analisis"</strong> untuk memulai deteksi 
                Alat Pelindung Diri secara otomatis menggunakan model YOLOv8.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 11. FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    Construction PPE Detection System &nbsp;|&nbsp; Powered by YOLOv8 and Streamlit
</div>
""", unsafe_allow_html=True)
