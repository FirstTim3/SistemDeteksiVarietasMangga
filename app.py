import PIL
import streamlit as st
import numpy as np

from ultralytics import YOLO

# Inisialisasi State
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
if "has_result" not in st.session_state:
    st.session_state["has_result"] = False

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
    img_pil = uploaded_img
    draw = PIL.ImageDraw.Draw(img_pil)

    # gunakan font lebih besar
    try:
        W, H = img.size
        font_size = max(20, W // 40)  # skala dinamis
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except:
        font = ImageFont.load_default()
    
    # font = PIL.ImageFont.load_default()
    names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{names[cls]} \nKeyakinan: {conf:.2f}"

        # Gambar kotak
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Latar teks
        text_size = draw.textbbox((0, 0), label, font=font)
        tw, th = text_size[2] - text_size[0], text_size[3] - text_size[1]
        
        pad = 4
        tx = x1 + pad
        ty = y1 + pad

                    #X          #Y          #W         #H                
        text_bg = [tx - pad, ty - pad, tx + tw + pad, ty + th + pad]
        draw.rectangle(text_bg, fill=box_color)
        draw.text((tx, ty), label, fill=text_color, font=font)
    return img_pil

# reset state ketika slider diubah
def invalidate_result():
    st.session_state["has_result"] = False

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
st.info("Unggah gambar terlebih dahulu dengan memilih Browse files")
st.info("Lalu tekan tombol Deteksi Objek untuk melakukan deteksi")
st.caption("File yang didukung: JPG/JPEG/PNG (MAX 200 MB)")

# Membuat form untuk mengunggah gambar
source_img = st.file_uploader(
        label="Pilih Gambar", 
        type=("jpg", "jpeg", "png"),
        help="Tarik & lepas gambar di area atau klik ‘Browse files’.",
        key=st.session_state["file_uploader_key"],
        on_change=invalidate_result
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

        # Menampilkan gambar
        st.image(uploaded_img,
                    caption="Gambar yang diunggah",
                    use_container_width=True
                    )
    except Exception as e:
        st.toast("Terjadi kesalahan saat membaca gambar.", icon="⚠️")
    except NameError:
        pass

if st.button("Deteksi Objek", type="primary", use_container_width=True):
    # Melakukan pengecekan apakah gambar sudah diunggah
    if source_img:
        # Melakukan prediksi
        try:
            prediction = model.predict(
                uploaded_img,
                conf=0.5
            )

            # Mendapatkan data hasil deteksi
            boxes = prediction[0].boxes

            # Membuat bounding box, label dan confidence score pada gambar
            prediction_plotted = draw_detection(prediction[0])
            
            st.image(
                prediction_plotted,
                caption="Hasil Deteksi",
                use_container_width=True
            )
        
            st.write(f'Jumlah Objek terdeteksi: {len(boxes)}')

            # Menampilkan detail hasil deteksi
            try:
                with st.expander("Hasil Deteksi (DESC)", expanded=True):
                    for i, box in enumerate(boxes):
                        names = prediction[0].names
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{i+1}.\t Prediksi: **{names[cls]}** | Keyakinan : {conf:.2f}({(100 * conf):.0f}%)"
                        xywh = [f"{v:.2f}" for v in box.xywh[0].tolist()]
                        st.write(label)
                        st.write(f"Bounding Box (xywh): {xywh}")
                        st.write("")
                
                st.session_state["has_result"]=True
            except Exception as e:
                st.toast(e)
        except ValueError:
            st.toast("Error", icon="❌")
        except NameError:
            pass
    else:
        st.toast("Unggah gambar terlebih dahulu!", icon="❌")

if st.session_state["has_result"]:
    if st.button("RESET", type="primary", use_container_width=True):
        # kosongkan status hasil
        st.session_state["has_result"] = False
        # remount uploader agar file hilang dari UI
        st.session_state["file_uploader_key"] += 1
        st.rerun()
