"""
streamlit_app.py

PPE Detection System - Aplikasi Streamlit dengan dua mode deteksi:
    1. Upload Gambar
    2. Kamera Real-Time (streamlit-webrtc)

Menggunakan model YOLOv8 (Ultralytics) yang sama untuk kedua mode.

Refactor total (2026) untuk production-ready di Streamlit Community Cloud:
    - Model dimuat sekali via @st.cache_resource dan tidak pernah reload.
    - Semua inferensi dipaksa berjalan di CPU (device="cpu").
    - Mode kamera menggunakan API streamlit-webrtc terbaru yang direkomendasikan
      (video_frame_callback berbasis fungsi), BUKAN kelas VideoProcessorBase
      yang sudah masuk fase deprecated menjelang v1.0.
    - Start/Stop kamera sepenuhnya diserahkan ke tombol bawaan streamlit-webrtc.
      Tidak ada session_state, desired_playing_state, atau while-loop yang
      memblokir thread utama Streamlit.
    - Error handling & logging ditambahkan di setiap titik rawan gagal:
      load model, prediksi gambar, prediksi frame, dan inisialisasi webrtc.

Dijalankan dengan:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ppe_detection")

# Redam log verbose dari library pihak ketiga yang sering menyebabkan
# noise (dan pada versi lama sempat memicu banyak retry/ICE log).
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# ----------------------------------------------------------------------
# KONFIGURASI GLOBAL
# ----------------------------------------------------------------------
MODEL_PATH = "weights/best.pt"
INFERENCE_DEVICE = "cpu"  # Streamlit Community Cloud tidak menyediakan GPU

# STUN server publik agar koneksi video via browser dapat menembus
# NAT/firewall pada jaringan tempat aplikasi diakses.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(
    page_title="PPE Detection",
    page_icon="🦺",
    layout="wide",
)


# ----------------------------------------------------------------------
# PEMUATAN MODEL (SATU KALI, DI-CACHE, TIDAK PERNAH RELOAD)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_model(model_path: str = MODEL_PATH) -> Optional[YOLO]:
    """
    Memuat model YOLOv8 hasil training sekali saja untuk seluruh siklus
    hidup aplikasi.

    @st.cache_resource memastikan objek model (yang tidak bisa/boleh
    di-hash) disimpan di memori proses dan dipakai ulang oleh setiap
    session, bukan dimuat ulang setiap kali script Streamlit dieksekusi.

    Returns:
        Instance YOLO yang siap dipakai, atau None jika gagal dimuat.
    """
    try:
        model = YOLO(model_path)
        model.to(INFERENCE_DEVICE)
        logger.info("Model berhasil dimuat dari %s", model_path)
        return model
    except FileNotFoundError:
        logger.exception("File model tidak ditemukan: %s", model_path)
        return None
    except Exception:  # noqa: BLE001 - ingin menangkap semua error load model
        logger.exception("Gagal memuat model dari %s", model_path)
        return None


# ----------------------------------------------------------------------
# FUNGSI INFERENSI
# ----------------------------------------------------------------------
def predict_image(model: YOLO, image: Image.Image, confidence: float):
    """
    Menjalankan inferensi YOLOv8 pada gambar hasil upload.

    Gambar dikonversi langsung ke array BGR di memori (tanpa file
    sementara di disk) sebelum diteruskan ke model.

    Returns:
        Objek hasil deteksi Ultralytics (Results), atau None jika gagal.
    """
    try:
        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(
            source=image_bgr,
            conf=confidence,
            device=INFERENCE_DEVICE,
            save=False,
            verbose=False,
        )
        return results[0]
    except Exception:  # noqa: BLE001
        logger.exception("Gagal menjalankan prediksi pada gambar upload")
        return None


def predict_frame(model: YOLO, frame_bgr: np.ndarray, confidence: float):
    """
    Menjalankan inferensi YOLOv8 pada satu frame webcam (array BGR).

    Dipanggil berulang kali oleh callback webrtc untuk setiap frame yang
    diterima secara real-time. Fungsi ini TIDAK BOLEH memanggil st.*
    karena dieksekusi pada thread/asyncio loop milik aiortc, bukan thread
    utama Streamlit.

    Returns:
        Objek hasil deteksi Ultralytics (Results), atau None jika gagal.
    """
    try:
        results = model.predict(
            source=frame_bgr,
            conf=confidence,
            device=INFERENCE_DEVICE,
            save=False,
            verbose=False,
        )
        return results[0]
    except Exception:  # noqa: BLE001
        logger.exception("Gagal menjalankan prediksi pada frame webcam")
        return None


def draw_results(result) -> np.ndarray:
    """
    Menggambar bounding box, nama kelas, dan confidence score dari hasil
    deteksi menggunakan utilitas plot() bawaan Ultralytics.

    Catatan: result.plot() selalu mengembalikan array berformat BGR
    (konvensi OpenCV), bukan RGB.
    """
    return result.plot()


def count_detections(result, model: YOLO) -> dict:
    """Menghitung jumlah objek terdeteksi per kelas dari satu hasil inferensi."""
    if result is None:
        return {}
    names = model.names
    counts: dict = {}
    for box in result.boxes:
        cls = int(box.cls[0])
        label = names[cls]
        counts[label] = counts.get(label, 0) + 1
    return counts


def render_detection_summary(counts: dict) -> None:
    """Menampilkan ringkasan jumlah deteksi per kelas APD di thread utama."""
    st.subheader("Detection Summary")
    if not counts:
        st.info("Tidak ada objek yang terdeteksi.")
        return
    for label, total in counts.items():
        st.write(f"**{label}** : {total}")


# ----------------------------------------------------------------------
# STATE BERSAMA UNTUK MODE KAMERA (THREAD-SAFE)
# ----------------------------------------------------------------------
class SharedFrameState:
    """
    Wadah state ringan yang dibagi antara:
      - thread utama Streamlit (membaca confidence dari slider, menampilkan
        detection summary), dan
      - thread/asyncio loop aiortc (menulis hasil deteksi terbaru di
        dalam callback video).

    Sebuah threading.Lock dipakai HANYA di sini karena memang dibutuhkan
    untuk mencegah race condition saat kedua sisi mengakses field yang
    sama secara bersamaan. Tidak ada thread tambahan yang dibuat secara
    manual - lock ini murni melindungi data, bukan mengatur alur eksekusi.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.confidence: float = 0.5
        self.detection_counts: dict = {}

    def set_confidence(self, confidence: float) -> None:
        with self.lock:
            self.confidence = confidence

    def get_confidence(self) -> float:
        with self.lock:
            return self.confidence

    def set_counts(self, counts: dict) -> None:
        with self.lock:
            self.detection_counts = counts

    def get_counts(self) -> dict:
        with self.lock:
            return dict(self.detection_counts)


def get_shared_state() -> SharedFrameState:
    """
    Mengambil (atau membuat sekali) SharedFrameState untuk session Streamlit
    saat ini. Disimpan di session_state (bukan cache_resource) karena state
    ini spesifik per pengguna/tab, bukan sesuatu yang harus dibagi ke semua
    pengguna aplikasi seperti model.
    """
    if "shared_frame_state" not in st.session_state:
        st.session_state.shared_frame_state = SharedFrameState()
    return st.session_state.shared_frame_state


def make_video_frame_callback(model: YOLO, shared_state: SharedFrameState):
    """
    Membuat fungsi callback video untuk streamlit-webrtc.

    Menggunakan API function-based (video_frame_callback) yang merupakan
    cara yang direkomendasikan pada streamlit-webrtc versi terbaru,
    menggantikan pola class-based VideoProcessorBase yang berstatus
    deprecated menjelang rilis v1.0.

    PENTING: fungsi ini berjalan di asyncio loop milik aiortc, BUKAN thread
    utama Streamlit, sehingga TIDAK BOLEH memanggil st.* di dalamnya.
    Semua komunikasi ke thread utama dilakukan lewat SharedFrameState
    yang dilindungi lock.
    """

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            confidence = shared_state.get_confidence()

            result = predict_frame(model, img, confidence)
            if result is None:
                # Gagal memprediksi frame ini: kembalikan frame asli agar
                # video tidak berhenti / freeze, cukup lewati anotasi.
                return frame

            annotated = draw_results(result)
            counts = count_detections(result, model)
            shared_state.set_counts(counts)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception:  # noqa: BLE001
            # Callback TIDAK BOLEH melempar exception ke aiortc, karena
            # dapat merusak koneksi WebRTC yang sedang berjalan. Cukup log
            # dan kembalikan frame asli tanpa anotasi.
            logger.exception("Error di dalam video_frame_callback")
            return frame

    return video_frame_callback


# ----------------------------------------------------------------------
# MODE 1: UPLOAD GAMBAR
# ----------------------------------------------------------------------
def upload_mode(model: YOLO, confidence: float) -> None:
    """Menangani alur upload gambar -> prediksi -> tampilkan hasil."""
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file)
    except Exception:  # noqa: BLE001
        logger.exception("Gagal membuka file gambar yang diupload")
        st.error("Gagal membuka file gambar. Pastikan file tidak rusak dan berformat JPG/JPEG/PNG.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    result = predict_image(model, image, confidence)
    if result is None:
        st.error("Gagal menjalankan deteksi pada gambar ini. Coba gambar lain atau ulangi.")
        return

    plotted = draw_results(result)

    with col2:
        st.subheader("Detection Result")
        # result.plot() mengembalikan array BGR, sedangkan st.image() secara
        # default mengasumsikan RGB. channels="BGR" mencegah warna tertukar.
        st.image(plotted, use_container_width=True, channels="BGR")

    counts = count_detections(result, model)
    render_detection_summary(counts)


# ----------------------------------------------------------------------
# MODE 2: KAMERA REAL-TIME
# ----------------------------------------------------------------------
def camera_mode(model: YOLO, confidence: float) -> None:
    """
    Menangani alur deteksi real-time via webcam menggunakan streamlit-webrtc.

    Catatan desain penting:
      - Tidak ada session_state yang dipakai untuk mengontrol Start/Stop.
        Tombol START/STOP bawaan streamlit-webrtc yang dipakai sepenuhnya.
      - Tidak ada parameter desired_playing_state, karena parameter ini
        sering memicu rerun terus-menerus dan konflik state.
      - Tidak ada `while ctx.state.playing: ...` yang memblokir thread utama.
      - Tidak ada polling/thread tambahan; detection summary ditampilkan
        berdasarkan state terakhir yang tersimpan dan diperbarui setiap
        Streamlit melakukan rerun (misalnya saat slider confidence digeser
        atau tombol refresh ditekan).
    """
    st.subheader("Video Real-Time")

    shared_state = get_shared_state()
    shared_state.set_confidence(confidence)

    callback = make_video_frame_callback(model, shared_state)

    try:
        ctx = webrtc_streamer(
            key="ppe-camera-realtime",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=callback,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Gagal menginisialisasi webrtc_streamer")
        st.error(
            "Tidak dapat memulai koneksi kamera. Periksa izin kamera di browser "
            "dan koneksi internet Anda, lalu muat ulang halaman."
        )
        return

    st.caption(
        "Gunakan tombol START/STOP di atas untuk mengaktifkan atau "
        "menghentikan kamera."
    )

    st.markdown("---")

    if st.button("🔄 Refresh Detection Summary"):
        st.rerun()

    if ctx.state.playing:
        counts = shared_state.get_counts()
        render_detection_summary(counts)
    else:
        st.subheader("Detection Summary")
        st.info("Kamera belum aktif. Klik tombol START untuk memulai deteksi.")


# ----------------------------------------------------------------------
# HALAMAN UTAMA
# ----------------------------------------------------------------------
def main() -> None:
    """Entry point aplikasi Streamlit."""
    st.title("🦺 PPE Detection System")
    st.write("Deteksi Penggunaan APD Menggunakan YOLOv8")

    model = load_model()
    if model is None:
        st.error(
            f"Model tidak dapat dimuat dari `{MODEL_PATH}`. "
            "Pastikan file model tersedia di path tersebut, lalu muat ulang halaman."
        )
        st.stop()

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
