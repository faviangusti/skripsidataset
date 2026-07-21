"""
app.py

Aplikasi web deteksi Alat Pelindung Diri (APD) secara real-time
menggunakan model YOLOv8 (Ultralytics) dan kamera webcam laptop.

Dijalankan dengan:
    streamlit run app.py
"""

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ----------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi APD Real-Time",
    page_icon="🦺",
    layout="wide",
)

MODEL_PATH = "weights/best.pt"


# ----------------------------------------------------------------------
# PEMUATAN MODEL (HANYA SEKALI, DI-CACHE OLEH STREAMLIT)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_model(model_path: str) -> YOLO:
    """
    Memuat model YOLOv8 hasil training.

    Menggunakan st.cache_resource (bukan st.cache yang sudah deprecated,
    dan bukan pula st.cache_data karena objek model bukan objek data biasa
    yang aman untuk di-serialize/copy). Dekorator ini memastikan proses
    loading model hanya berjalan SATU KALI selama siklus hidup aplikasi,
    walaupun Streamlit menjalankan ulang seluruh script pada setiap
    interaksi pengguna (klik tombol, geser slider, dsb). Ini adalah kunci
    utama optimasi performa dibanding versi upload gambar sebelumnya, di
    mana model berpotensi dimuat ulang setiap kali script dieksekusi ulang.
    """
    return YOLO(model_path)


model = load_model(MODEL_PATH)


# ----------------------------------------------------------------------
# KELAS VIDEO PROCESSOR
# ----------------------------------------------------------------------
class YOLOVideoProcessor(VideoProcessorBase):
    """
    Menangani setiap frame video webcam secara real-time.

    Method recv() dipanggil otomatis oleh streamlit-webrtc pada THREAD
    TERPISAH dari thread utama Streamlit, untuk setiap frame yang
    diterima dari browser pengguna melalui protokol WebRTC.

    Model YOLO TIDAK dimuat di dalam kelas ini, melainkan diterima
    sebagai parameter dari luar (dependency injection). Dengan begitu,
    objek model yang sudah di-cache oleh Streamlit (load_model) dapat
    dipakai ulang, bukan dimuat ulang setiap frame diproses.
    """

    def __init__(self, model: YOLO, confidence_threshold: float = 0.5):
        self.model = model
        # Disimpan sebagai atribut instance agar nilainya dapat diperbarui
        # secara LIVE dari luar kelas (lihat blok kode slider di bawah),
        # tanpa perlu menghentikan dan memulai ulang koneksi webcam.
        self.confidence_threshold = confidence_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Konversi frame dari format av.VideoFrame (WebRTC) ke array
        # numpy berformat BGR, yaitu format standar yang digunakan OpenCV
        img = frame.to_ndarray(format="bgr24")

        # Jalankan satu kali inferensi YOLOv8 pada frame saat ini.
        # verbose=False mencegah log konsol membanjir pada tiap frame.
        results = self.model.predict(
            img, conf=self.confidence_threshold, verbose=False
        )[0]

        annotated_img = self._draw_detections(img, results)

        # Kembalikan frame yang sudah dianotasi dalam format av.VideoFrame
        # agar dapat ditampilkan kembali oleh komponen streamlit-webrtc
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    def _draw_detections(self, img: np.ndarray, results) -> np.ndarray:
        """
        Menggambar bounding box, nama kelas, dan confidence score
        untuk setiap objek yang terdeteksi pada satu frame.
        """
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[cls_id]

            label_text = f"{class_name} {confidence:.2f}"

            # Gambar kotak deteksi (bounding box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Hitung ukuran teks agar latar belakang label pas ukurannya
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Gambar latar belakang label agar teks tetap mudah dibaca
            # di atas warna latar/gambar apapun
            cv2.rectangle(
                img,
                (x1, max(y1 - text_h - 10, 0)),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1,
            )

            # Tampilkan teks nama kelas dan confidence score
            cv2.putText(
                img, label_text, (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

        return img


# ----------------------------------------------------------------------
# ANTARMUKA UTAMA
# ----------------------------------------------------------------------
st.title("Deteksi Alat Pelindung Diri (APD) Real-Time")
st.markdown(
    "Aplikasi ini mendeteksi Alat Pelindung Diri (APD) secara real-time "
    "menggunakan model YOLOv8 dan kamera webcam laptop."
)

# ------------------------------------------------------------------
# SIDEBAR: PENGATURAN
# ------------------------------------------------------------------
st.sidebar.header("Pengaturan Deteksi")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Ambang batas kepercayaan minimum agar suatu deteksi ditampilkan",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model dimuat satu kali menggunakan st.cache_resource, sehingga "
    "perubahan slider TIDAK menyebabkan model dimuat ulang maupun "
    "koneksi kamera terputus."
)

# ------------------------------------------------------------------
# KONTROL START / STOP KAMERA
# ------------------------------------------------------------------
# st.session_state digunakan agar status kamera (aktif/tidak aktif)
# tetap konsisten antar rerun script, karena Streamlit menjalankan
# ulang SELURUH script setiap kali ada interaksi pengguna
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

col_start, col_stop = st.columns(2)
if col_start.button("Start Camera", use_container_width=True):
    st.session_state.camera_running = True
if col_stop.button("Stop Camera", use_container_width=True):
    st.session_state.camera_running = False

# ------------------------------------------------------------------
# KONFIGURASI WEBRTC
# ------------------------------------------------------------------
# STUN server publik digunakan agar koneksi video via browser dapat
# menembus NAT/firewall pada jaringan tempat aplikasi diakses
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ------------------------------------------------------------------
# STREAMING WEBCAM REAL-TIME
# ------------------------------------------------------------------
ctx = webrtc_streamer(
    key="ppe-detection-realtime",
    # desired_playing_state memungkinkan status kamera dikontrol secara
    # programatik melalui tombol Start/Stop kustom di atas, alih-alih
    # hanya mengandalkan tombol bawaan komponen streamlit-webrtc
    desired_playing_state=st.session_state.camera_running,
    video_processor_factory=lambda: YOLOVideoProcessor(
        model=model, confidence_threshold=confidence_threshold
    ),
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Perbarui confidence threshold secara LIVE tanpa menghentikan stream,
# dengan mengubah atribut pada instance video_processor yang SEDANG
# berjalan secara langsung (bukan membuat ulang koneksi webcam)
if ctx.video_processor:
    ctx.video_processor.confidence_threshold = confidence_threshold

st.markdown(
    """
    ---
    **Petunjuk penggunaan:**
    1. Klik **Start Camera** untuk mengaktifkan webcam.
    2. Izinkan akses kamera pada browser ketika diminta.
    3. Geser slider *Confidence Threshold* untuk menyesuaikan sensitivitas deteksi secara langsung.
    4. Klik **Stop Camera** untuk menghentikan streaming.
    """
)
