import PIL
import streamlit as st
import numpy as np

from ultralytics import YOLO

# Validasi gambar
def is_image_file_valid(src_img):
    try:
        with PIL.Image.open(src_img) as img:
            img.verify()
        with PIL.Image.open(src_img) as img:
            img.load()
        return True
    except Exception:
        st.toast("Gambar tidak valid. Coba gambar yang lain.", icon="❌")
        return False

# Gambar plot
def draw_detection(result, box_color=(0, 114, 255), text_color=(255, 255, 255)):
    # Ambil gambar asli (BGR -> RGB)
    img_rgb = result.orig_img[:, :, ::-1]  
    img_pil = PIL.Image.fromarray(img_rgb)
    draw = PIL.ImageDraw.Draw(img_pil)

    # Gunakan font bawaan
    font = PIL.ImageFont.load_default()
    names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"

        # Gambar kotak
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Latar teks
        text_size = draw.textbbox((0, 0), label, font=font)
        tw, th = text_size[2] - text_size[0], text_size[3] - text_size[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=box_color)
        draw.text((x1, y1 - th), label, fill=text_color, font=font)

    return img_pil


# Load Model
try:
    model = YOLO("best.pt")
except:
    st.error("Gagal mengakses model.")

# Set Page Title
st.set_page_config(
    page_title="Sistem Deteksi Objek",  # Setting page title
    layout="wide",      # Setting layout to wide
)

# Set Page Header
st.title("Sistem Deteksi Varietas Pohon Mangga Berdasarkan Citra Daun")

# Menambahkan instruksi penggunaan aplikasi
st.info("Unggah gambar terlebih dahulu dengan memilih :blue[Browse files]")
st.info("Lalu tekan tombol :blue[Deteksi Objek] untuk melakukan deteksi")
st.caption("File yang didukung: JPG/JPEG/PNG (MAX 200 MB)")

# Membuat form untuk mengunggah gambar
source_img = st.file_uploader(
        label="Pilih Gambar", 
        type=("jpg", "jpeg", "png"),
        help="Tarik & lepas gambar di area atau klik ‘Browse files’."
    )

# Melakukan modifikasi pada tampilan file uploader
st.markdown("""
<style>
/* Menyembunyikan baris "Limit 200MB per file • JPG, JPEG, PNG" di dalam dropzone */
  div[data-testid="stFileUploaderDropzoneInstructions"] div > span:nth-of-type(2) {
  display: none;
    }        
</style>
""", unsafe_allow_html=True)

# Cek gambar
if source_img is not None and is_image_file_valid(source_img):
    try:
        # Membuka gambar yg diunggah
        uploaded_img = PIL.Image.open(source_img)
        img = np.array(uploaded_img.convert('RGB'))
        image_width, image_height = uploaded_img.size

        # Menampilkan gambar
        st.image(source_img,
                    caption="Gambar yang diunggah",
                    use_container_width=True
                    )
    except Exception as e:
        st.toast("Terjadi kesalahan saat membaca gambar.", icon="⚠️")
    except NameError:
        pass

confidence = st.slider(
    "Select Model Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.01
)

if st.button('Deteksi Objek'):
    # Menyimpan state, agar prediksi hanya dilakukan saat tombol ditekan
    # Menghindari prediksi berulang-ulang saat slider diubah
    st.session_state["trigger_predict"] = True

    # Melakukan pengecekan apakah gambar sudah diunggah
    if source_img:
        # Melakukan pengecekan terhadap state dan menset ulang state jika belum diset
        if st.session_state.get("trigger_predict", False):
            # Melakukan prediksi
            try:
                prediction = model.predict(
                    uploaded_img,
                    conf=confidence
                )

                # Mendapatkan data hasil deteksi
                boxes = prediction[0].boxes

                # Membuat bounding box, label dan confidence score pada gambar
                prediction_plotted = draw_detection(prediction[0])
                #prediction_plotted = prediction[0].plot()[:, :, ::-1]
                
                # Menampilkan gambar hasil deteksi
                st.image(
                    prediction_plotted,
                    caption="Hasil Deteksi",
                    use_container_width=True
                )
                # Menampilkan detail hasil deteksi
                try:
                    # Menampilkan jumlah objek terdeteksi
                    st.write(f'Jumlah Objek terdeteksi: {len(boxes)}')

                    # Menampilkan bounding box objek terdeteksi
                    with st.expander("Bounding Box (xywh)"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as e:
                    st.toast(e)
            except NameError:
                pass

        # Mereset state
        st.session_state["trigger_predict"] = False
    else:
        st.toast("Unggah gambar terlebih dahulu!", icon="❌")






