"""
streamlit_app.py

PPE Detection System - Aplikasi Streamlit dengan dua mode deteksi:
1. Upload Gambar
2. Kamera Laptop (Real-Time)

Menggunakan model YOLOv8 (Ultralytics) yang sama untuk kedua mode.
Merupakan hasil refactor dari aplikasi upload-gambar yang sudah ada
sebelumnya, dengan penambahan mode kamera real-time via streamlit-webrtc.

Dijalankan dengan:
    streamlit run streamlit_app.py
"""

import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------------------------------------------------
# KONFIGURASI GLOBAL
# ----------------------------------------------------------------------
MODEL_PATH = "weights/best.pt"

# STUN server publik agar koneksi video via browser dapat menembus
# NAT/firewall pada jaringan tempat aplikasi diakses
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(
    page_title="PPE Detection",
    page_icon="🦺",
    layout="wide",
)


# ----------------------------------------------------------------------
# PEMUATAN MODEL (SATU KALI, DI-CACHE)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    Memuat model YOLOv8 hasil training.

    PERBAIKAN dari kode asli: sebelumnya model dimuat via baris global
    `model = YOLO("weights/best.pt")` tanpa cache, sehingga berpotensi
    dimuat ulang setiap kali script Streamlit dieksekusi ulang (setiap
    interaksi pengguna, misalnya geser slider). Dengan @st.cache_resource,
    model HANYA dimuat sekali selama siklus hidup aplikasi.
    """
    return YOLO(model_path)


# ----------------------------------------------------------------------
# FUNGSI INFERENSI
# ----------------------------------------------------------------------
def predict_image(model: YOLO, image: Image.Image, confidence: float):
    """
    Menjalankan inferensi YOLOv8 pada gambar hasil upload.

    PERBAIKAN dari kode asli: sebelumnya gambar disimpan dahulu ke file
    sementara (tempfile.NamedTemporaryFile) sebelum diproses. Cara ini
    memiliki dua masalah:
    1. Menambah overhead I/O disk yang sebenarnya tidak perlu, karena
       YOLOv8 bisa menerima array gambar langsung dari memori.
    2. Bug memory/disk leak: parameter delete=False membuat file
       sementara tersebut TIDAK PERNAH dihapus, sehingga setiap upload
       gambar meninggalkan file sisa permanen di disk.

    Fungsi ini menghilangkan kebutuhan file sementara sepenuhnya dengan
    mengonversi gambar PIL langsung menjadi array BGR di memori (format
    yang konsisten dengan konvensi OpenCV/Ultralytics).
    """
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    results = model.predict(source=image_bgr, conf=confidence,device="cpu", save=False, verbose=False)
    return results[0]


def predict_frame(model: YOLO, frame_bgr: np.ndarray, confidence: float):
    """
    Menjalankan inferensi YOLOv8 pada satu frame webcam (array BGR).
    Dipanggil berulang kali oleh video processor untuk setiap frame
    yang diterima secara real-time.
    """
    results = model.predict(source=frame_bgr, conf=confidence,device="cpu", save=False, verbose=False)
    return results[0]


def draw_results(result) -> np.ndarray:
    """
    Menggambar bounding box, nama kelas, dan confidence score dari hasil
    deteksi menggunakan utilitas plot() bawaan Ultralytics.

    Catatan penting: result.plot() selalu mengembalikan array dengan
    urutan channel BGR (konvensi OpenCV), bukan RGB. Ini harus
    diperhitungkan saat menampilkannya melalui st.image() -- lihat
    penjelasan bug pada fungsi upload_mode().
    """
    return result.plot()


def count_detections(result, model: YOLO) -> dict:
    """Menghitung jumlah objek terdeteksi per kelas dari satu hasil inferensi."""
    names = model.names
    counts = {}
    for box in result.boxes:
        cls = int(box.cls[0])
        label = names[cls]
        counts[label] = counts.get(label, 0) + 1
    return counts


def show_detection_summary(counts: dict) -> None:
    """Menampilkan ringkasan jumlah deteksi per kelas APD."""
    st.subheader("Detection Summary")
    if len(counts) == 0:
        st.info("Tidak ada objek yang terdeteksi.")
        return
    for label, total in counts.items():
        st.write(f"**{label}** : {total}")


# ----------------------------------------------------------------------
# VIDEO PROCESSOR UNTUK MODE KAMERA
# ----------------------------------------------------------------------
class YOLOVideoProcessor(VideoProcessorBase):
    """
    Menangani setiap frame webcam secara real-time.

    Method recv() dipanggil otomatis oleh streamlit-webrtc pada THREAD
    TERPISAH dari thread utama Streamlit. Model YOLO diterima sebagai
    parameter (bukan dimuat di dalam kelas ini) agar objek model yang
    sudah di-cache oleh load_model() dapat dipakai ulang tanpa reload.

    self.lock digunakan karena self.detection_counts ditulis oleh thread
    recv() dan dibaca oleh thread utama Streamlit (di camera_mode()).
    Tanpa lock, race condition berpotensi terjadi saat kedua thread
    mengakses dictionary yang sama secara bersamaan.
    """

    def __init__(self, model: YOLO, confidence_threshold: float = 0.5):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.detection_counts: dict = {}
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        result = predict_frame(self.model, img, self.confidence_threshold)
        annotated = draw_results(result)
        counts = count_detections(result, self.model)

        with self.lock:
            self.detection_counts = counts

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ----------------------------------------------------------------------
# MODE 1: UPLOAD GAMBAR
# ----------------------------------------------------------------------
def upload_mode(model: YOLO, confidence: float) -> None:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    result = predict_image(model, image, confidence)
    plotted = draw_results(result)

    with col2:
        st.subheader("Detection Result")
        # PERBAIKAN dari kode asli: result.plot() mengembalikan array BGR,
        # sedangkan st.image() secara default mengasumsikan array numpy
        # berformat RGB. Tanpa parameter channels="BGR", warna merah dan
        # biru pada hasil deteksi akan tertukar (bug tampilan pada kode asli).
        st.image(plotted, use_container_width=True, channels="BGR")

    counts = count_detections(result, model)
    show_detection_summary(counts)


# ----------------------------------------------------------------------
# MODE 2: KAMERA REAL-TIME
# ----------------------------------------------------------------------
def camera_mode(model: YOLO, confidence: float) -> None:
    st.subheader("Video Real-Time")

    # session_state menjaga status kamera (aktif/tidak) tetap konsisten
    # antar rerun script, karena Streamlit menjalankan ulang seluruh
    # script setiap kali ada interaksi pengguna
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    col_start, col_stop = st.columns(2)
    if col_start.button("Start Camera", use_container_width=True):
        st.session_state.camera_running = True
    if col_stop.button("Stop Camera", use_container_width=True):
        st.session_state.camera_running = False

    ctx = webrtc_streamer(
        key="ppe-camera-realtime",
        desired_playing_state=st.session_state.camera_running,
        video_processor_factory=lambda: YOLOVideoProcessor(
            model=model, confidence_threshold=confidence
        ),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Perbarui confidence threshold secara LIVE (slider yang sama dengan
    # mode upload) tanpa perlu menghentikan dan memulai ulang stream
    if ctx.video_processor:
        ctx.video_processor.confidence_threshold = confidence

    st.subheader("Detection Summary")
    summary_placeholder = st.empty()

    if ctx.state.playing and ctx.video_processor:
        with ctx.video_processor.lock:
            counts = ctx.video_processor.detection_counts.copy()

        with summary_placeholder.container():
            if not counts:
                st.info("Tidak ada objek yang terdeteksi.")
            else:
                for label, total in counts.items():
                    st.write(f"**{label}** : {total}")
    else:
        summary_placeholder.info(
            "Kamera belum aktif. Klik Start Camera untuk memulai."
        )



# ----------------------------------------------------------------------
# HALAMAN UTAMA
# ----------------------------------------------------------------------
def main() -> None:
    st.title("🦺 PPE Detection System")
    st.write("Deteksi Penggunaan APD Menggunakan YOLOv8")

    model = load_model()

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    mode = st.radio(
        "Pilih Mode",
        options=["Upload Gambar", "Kamera Real-Time"],
        horizontal=True,
    )

    st.markdown("---")

    if mode == "Upload Gambar":
        upload_mode(model, confidence)
    else:
        camera_mode(model, confidence)


if __name__ == "__main__":
    main()
