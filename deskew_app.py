import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import img2pdf
import os

def detect_skew_angle_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0

    angles = []
    for rho, theta in lines[:,0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    if len(angles) == 0:
        return 0
    return np.median(angles)

def deskew_image_hough(pil_image):
    image = np.array(pil_image)
    angle = detect_skew_angle_hough(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated), angle

# --- Giao diện Streamlit ---
st.set_page_config(page_title="Deskew PDF App", page_icon="📄", layout="centered")

st.markdown("""
<style>
.main > div {max-width: 900px; margin: auto;}
h1 {color: #4B8BBE; font-weight: bold; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
h3 {color: #306998;}
.stButton>button {background-color: #4B8BBE; color: white; border-radius: 8px; padding: 8px 20px;}
.stButton>button:hover {background-color: #306998;}
</style>
""", unsafe_allow_html=True)

st.title("📄 Ứng dụng hiệu chỉnh PDF")
st.title("📄 Tác giả: Nguyễn Hoàng Quy")
st.markdown("""
    Upload file PDF cần chỉnh sửa. Ứng dụng sẽ tự động phát hiện và xoay các trang PDF để căn thẳng lại.
""")

uploaded_file = st.file_uploader("Chọn file PDF để xử lý", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Đang xử lý file... Vui lòng chờ chút nhé! ⏳"):
        # Lưu tạm file upload
        input_pdf_path = "temp_input.pdf"
        with open(input_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Tạo thư mục lưu ảnh đã chỉnh
        os.makedirs("deskewed_pages", exist_ok=True)
        pages = convert_from_path(input_pdf_path, dpi=300)

        deskewed_images = []
        angles = []

        for i, page in enumerate(pages):
            deskewed_img, angle = deskew_image_hough(page)
            angles.append(angle)
            path = f"deskewed_pages/page_{i:03}.jpg"
            deskewed_img.save(path, "JPEG")
            deskewed_images.append(path)

        # Tạo file PDF mới
        base_name = uploaded_file.name[:-4]  # bỏ .pdf
        new_filename = base_name + "_Da_Chinh.pdf"
        with open(new_filename, "wb") as f_out:
            f_out.write(img2pdf.convert(deskewed_images))

    st.success(f"✅ Hoàn tất xử lý! File PDF đã chỉnh được lưu tên: **{new_filename}**")

    st.write(angles)

    # Nút tải file
    with open(new_filename, "rb") as f:
        st.download_button(
            label="📥 Tải file PDF đã chỉnh",
            data=f.read(),
            file_name=new_filename,
            mime="application/pdf"
        )
else:
    st.info("👉 Vui lòng tải lên file PDF để bắt đầu.")
