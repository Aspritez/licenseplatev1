"""Thai license-plate deskew app for Streamlit Community Cloud."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from plate_detection import DATASET_CONFIG, DEFAULT_CONFIG, auto_detect_plate, detect_plate, order_points, validate_quadrilateral


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DISPLAY_WIDTH = 740
MAX_PROCESSING_DIMENSION = 1600
DETECTOR_SETTINGS_VERSION = 7
DEFAULT_PRESET = "ทดลอง dataset v23"
PRESETS: dict[str, dict[str, Any]] = {
    "ภาพใกล้": {"kernel": 7, "canny_low": 35, "canny_high": 110, "min_area": 350, "clahe": 2.0, "unsharp": 1.2, "description": "ค่าตั้งต้นที่ผ่านการลองกับภาพป้ายขนาดใหญ่: blur มากขึ้นเพื่อลดขอบรายละเอียดของรถ"},
    "ภาพปานกลาง": {"kernel": 7, "canny_low": 35, "canny_high": 110, "min_area": 150, "clahe": 2.5, "unsharp": 1.5, "description": "ค่าตั้งต้นที่ให้ผลสม่ำเสมอกับป้ายขนาดกลาง: ลด noise ก่อนหาเส้นขอบหลัก"},
    "ภาพไกล": {"kernel": 5, "canny_low": 20, "canny_high": 80, "min_area": 50, "clahe": 3.0, "unsharp": 2.0, "description": "เก็บขอบที่อ่อนของป้ายไกลด้วย Canny ที่ไวขึ้น แต่ยัง blur พอช่วยลดจุดรบกวน"},
    "ทดลอง dataset v23": {"kernel": 3, "canny_low": 20, "canny_high": 80, "min_area": 150, "clahe": 2.5, "unsharp": 1.5, "detector_config": DATASET_CONFIG, "description": "ค่าเริ่มต้น: ใช้ Blur 3, Canny 20/80 และเกณฑ์คะแนน/มุมจาก dataset v23 เมื่อกดค้นหาป้ายอัตโนมัติ"},
}


def initialise_state() -> None:
    for key, value in {"preset": DEFAULT_PRESET, "applied_preset": None, "image_token": None, "selected_points": [], "last_click_token": None}.items():
        st.session_state.setdefault(key, value)
    # Refresh preset state once after the detector settings change.
    if st.session_state.get("detector_settings_version") != DETECTOR_SETTINGS_VERSION:
        st.session_state["detector_settings_version"] = DETECTOR_SETTINGS_VERSION
        st.session_state["preset"] = DEFAULT_PRESET
        st.session_state["applied_preset"] = None


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        if key not in ("description", "detector_config"):
            st.session_state[key] = value
    st.session_state["applied_preset"] = name


def warp_plate(image_bgr: np.ndarray, ordered_points: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = ordered_points
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    output_width, output_height = max(2, round(width)), max(2, round(height))
    destination = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    # getPerspectiveTransform produces the 3×3 homography between the four
    # selected source points and this rectangular destination plane.
    homography = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), destination)
    return cv2.warpPerspective(image_bgr, homography, (output_width, output_height), flags=cv2.INTER_CUBIC)


def enhance_plate(image_bgr: np.ndarray, clahe_clip: float, unsharp_strength: float) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(lightness), channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    if unsharp_strength <= 0:
        return enhanced
    return cv2.addWeighted(enhanced, 1.0 + unsharp_strength, cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0), -unsharp_strength, 0)


def draw_points(image_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = image_bgr.copy()
    colours = [(0, 0, 255), (255, 180, 0), (0, 165, 255), (0, 255, 255)]
    labels = ["TL", "TR", "BR", "BL"]
    points = points.astype(np.int32)
    for index in range(4):
        cv2.line(result, tuple(points[index]), tuple(points[(index + 1) % 4]), (0, 220, 0), 2)
    for point, colour, label in zip(points, colours, labels):
        cv2.circle(result, tuple(point), 7, colour, -1)
        cv2.circle(result, tuple(point), 9, (255, 255, 255), 2)
        cv2.putText(result, label, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return result


def slider_with_info(label: str, minimum: Any, maximum: Any, step: Any, key: str, description: str) -> None:
    """Render an adjustment control with its purpose immediately beside it."""
    control_column, info_column = st.columns([3, 2])
    with control_column:
        st.slider(label, minimum, maximum, step=step, key=key, help=description)
    with info_column:
        st.info(description, icon="ℹ️")


def image_as_png(image_bgr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG")
    return buffer.getvalue()


def resize_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Downscale only oversized uploads and return scales back to the original."""
    original_height, original_width = image_bgr.shape[:2]
    longest_side = max(original_width, original_height)
    if longest_side <= MAX_PROCESSING_DIMENSION:
        return image_bgr, 1.0, 1.0

    resize_ratio = MAX_PROCESSING_DIMENSION / float(longest_side)
    resized_width = max(1, round(original_width * resize_ratio))
    resized_height = max(1, round(original_height * resize_ratio))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, original_width / resized_width, original_height / resized_height


st.set_page_config(page_title="Thai License Plate Deskew", page_icon="🚗", layout="wide")
initialise_state()
st.markdown("<style>.block-container { max-width: 1180px; padding-top: 2rem; }</style>", unsafe_allow_html=True)

st.title("🚗 ปรับมุมภาพป้ายทะเบียน")
st.caption("อัปโหลดรูป เลือกมุมป้าย 4 จุด แล้วดาวน์โหลดภาพป้ายที่ปรับมุมและเพิ่มความคมชัดแล้ว")
uploaded_file = st.file_uploader("อัปโหลดภาพรถ (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("อัปโหลดรูปภาพเพื่อเริ่มใช้งาน")
    st.stop()

file_bytes = uploaded_file.getvalue()
if len(file_bytes) > MAX_UPLOAD_BYTES:
    st.error("ไฟล์มีขนาดใหญ่เกิน 10 MB")
    st.stop()
try:
    original_rgb = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
except (OSError, ValueError) as error:
    st.error(f"ไม่สามารถเปิดรูปภาพนี้ได้: {error}")
    st.stop()

original_height, original_width = original_bgr.shape[:2]
processing_bgr, processing_to_original_x, processing_to_original_y = resize_for_detection(original_bgr)
processing_height, processing_width = processing_bgr.shape[:2]
image_token = hashlib.sha256(file_bytes).hexdigest()[:16]
if st.session_state["image_token"] != image_token:
    st.session_state["image_token"] = image_token
    st.session_state["selected_points"] = []
    st.session_state["last_click_token"] = None
st.info(f"ขนาดภาพต้นฉบับ: {original_width} × {original_height} พิกเซล — เลือกจุดตามลำดับใดก็ได้ ระบบเรียงมุมให้เอง")
if (processing_width, processing_height) != (original_width, original_height):
    st.caption(
        f"ค้นหาอัตโนมัติบนสำเนาขนาด {processing_width} × {processing_height} พิกเซล "
        f"(จำกัดด้านยาว {MAX_PROCESSING_DIMENSION} px) เพื่อให้ทำงานเร็วขึ้น; crop และไฟล์ดาวน์โหลดยังใช้ภาพต้นฉบับ"
    )

st.subheader("เลือกระยะของป้ายทะเบียน")
preset = st.selectbox("ระยะภาพ", list(PRESETS), key="preset")
detector_config = PRESETS[preset].get("detector_config", DEFAULT_CONFIG)
if st.session_state["applied_preset"] != preset:
    apply_preset(preset)
st.caption(PRESETS[preset]["description"])
st.caption("มี preset ทดลองจากชุด YOLO 456 ภาพ แยกตามกลุ่มต้นฉบับเป็นชุดปรับค่า / validation / test คะแนนวัดตำแหน่งกรอบ ไม่ได้วัดความแม่นของมุมทั้ง 4 ควรตรวจกรอบก่อน crop")

st.subheader("ปรับค่าก่อนค้นหาป้ายทะเบียน")
st.caption("ค่าเหล่านี้มีผลต่อการค้นหาอัตโนมัติและภาพผลลัพธ์ ใช้ค่า Preset เป็นจุดเริ่มต้น แล้วปรับเมื่อระบบเลือกป้ายไม่ตรง")
with st.container(border=True):
    slider_with_info("Gaussian blur", 1, 15, 2, "kernel", "ลด noise ก่อนหาเส้นขอบ: ค่าสูงช่วยภาพมีจุดรบกวนมาก แต่สูงเกินไปจะทำให้ขอบป้ายและตัวอักษรหาย")
    slider_with_info("Canny threshold ต่ำ", 10, 250, 5, "canny_low", "กำหนดความไวต่อเส้นขอบอ่อน: ลดค่านี้เมื่อป้ายเบลอหรือมืด แต่ค่าต่ำมากจะเก็บขอบของพื้นหลังเพิ่ม")
    slider_with_info("Canny threshold สูง", 20, 300, 5, "canny_high", "กำหนดเส้นขอบที่ชัดมาก: ลดค่านี้เมื่อขอบป้ายไม่คม เพิ่มค่านี้เมื่อต้องการลดเส้นรบกวน")
    slider_with_info("พื้นที่ต่ำสุดของกรอบป้าย", 50, 5000, 50, "min_area", "พื้นที่ขั้นต่ำอ้างอิงภาพ 1920×1080 และระบบจะสเกลตามความละเอียดภาพจริง: ใช้ค่าน้อยสำหรับป้ายไกล เพิ่มค่าสำหรับภาพใกล้เพื่อตัดวัตถุเล็ก")
    st.markdown("##### ปรับภาพหลัง crop")
    slider_with_info("CLAHE", 1.0, 5.0, 0.1, "clahe", "เพิ่ม contrast เฉพาะบริเวณ ช่วยให้ตัวอักษรบนป้ายที่มีแสงไม่สม่ำเสมอเด่นขึ้น โดยไม่มีผลต่อจุดที่ระบบตรวจจับ")
    slider_with_info("ความคมชัด", 0.0, 3.0, 0.1, "unsharp", "เร่งขอบตัวอักษรหลัง crop: เริ่มที่ 1.0–1.5 และอย่าปรับสูงเกินไป เพราะอาจเกิดขอบหลอกหรือ noise")

detect_column, undo_column, clear_column = st.columns(3)
detect_clicked = detect_column.button("🔍 ค้นหาป้ายอัตโนมัติ", width="stretch")
undo_clicked = undo_column.button("↩️ ลบจุดล่าสุด", width="stretch")
clear_clicked = clear_column.button("🗑️ เลือกใหม่", width="stretch")
if clear_clicked:
    st.session_state["selected_points"] = []
    st.rerun()
if undo_clicked and st.session_state["selected_points"]:
    st.session_state["selected_points"].pop()
    st.rerun()
if detect_clicked:
    with st.spinner("กำลังค้นหาป้ายทะเบียน..."):
        detection = detect_plate(processing_bgr, st.session_state["kernel"], st.session_state["canny_low"], st.session_state["canny_high"], st.session_state["min_area"], config=detector_config)
    detected = detection.corners
    if detected is None:
        st.warning("ยังไม่พบป้ายอัตโนมัติ กรุณาเลือก 4 จุดด้วยตนเอง")
    else:
        detected_in_original = detected * np.array([processing_to_original_x, processing_to_original_y], dtype=np.float32)
        st.session_state["selected_points"] = detected_in_original.astype(float).tolist()
        source = "ใช้ขอบป้ายจาก contour แรก" if detection.selected.corner_source == "direct" else "ค้นหาขอบป้ายซ้ำใน ROI"
        st.success(f"พบตำแหน่งที่เป็นไปได้แล้ว — คะแนน {detection.selected.score:.3f} · {source} ตรวจสอบหรือเลือกใหม่ได้")
    with st.expander("คะแนนบริเวณที่อาจเป็นป้ายทะเบียน", expanded=True):
        st.caption("คะแนนรวม 0–1 ใช้จัดอันดับ ไม่ใช่เปอร์เซ็นต์ความแม่นยำ เลือกอันดับสูงสุดที่ผ่านเกณฑ์และหามุมได้ครบ 4 จุด")
        feature_labels = ("Area", "Aspect ratio", "Rectangularity", "Solidity", "Approximation", "Edge density")
        st.caption("น้ำหนัก: " + " · ".join(f"{label} {weight:.0%}" for label, weight in zip(feature_labels, detector_config.weights)))
        rows = []
        for rank, candidate in enumerate(detection.candidates, 1):
            if rank > 20 and candidate is not detection.selected:
                continue
            rows.append({
                "อันดับ": rank,
                "เลือก": candidate is detection.selected,
                "ROI (x, y, w, h)": str(candidate.bbox),
                "คะแนนรวม": round(candidate.score, 3),
                **{name: round(value, 3) for name, value in candidate.scores.items()},
            })
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("ยังไม่มีบริเวณที่ผ่านเงื่อนไขพื้นฐาน")

display_width = min(MAX_DISPLAY_WIDTH, original_width)
display_height = max(1, round(original_height * display_width / original_width))
display_bgr = cv2.resize(original_bgr, (display_width, display_height), interpolation=cv2.INTER_AREA)
current_points = np.asarray(st.session_state["selected_points"], dtype=np.float32)
if len(current_points) == 4:
    displayed_points = current_points * np.array([display_width / original_width, display_height / original_height], dtype=np.float32)
    display_bgr = draw_points(display_bgr, displayed_points)
elif len(current_points):
    for index, point in enumerate(current_points):
        x, y = int(point[0] * display_width / original_width), int(point[1] * display_height / original_height)
        cv2.circle(display_bgr, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(display_bgr, str(index + 1), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

st.subheader(f"เลือกมุมป้ายทะเบียน: {len(current_points)}/4 จุด")
click = streamlit_image_coordinates(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB), width=display_width, key=f"plate_clicker_{image_token}", cursor="crosshair")
if click and click.get("unix_time") != st.session_state["last_click_token"]:
    st.session_state["last_click_token"] = click.get("unix_time")
    if len(st.session_state["selected_points"]) >= 4:
        st.warning("เลือกครบ 4 จุดแล้ว กด “เลือกใหม่” หรือ “ลบจุดล่าสุด” หากต้องการแก้ไข")
    else:
        x = float(click["x"]) * original_width / float(click["width"])
        y = float(click["y"]) * original_height / float(click["height"])
        candidate = np.asarray(st.session_state["selected_points"] + [[x, y]], dtype=np.float32)
        if len(candidate) > 1 and np.min(np.linalg.norm(candidate[:-1] - candidate[-1], axis=1)) < 8.0:
            st.warning("จุดใหม่นี้ซ้ำหรือใกล้กับจุดเดิมเกินไป")
        else:
            st.session_state["selected_points"].append([x, y])
            st.rerun()

points = np.asarray(st.session_state["selected_points"], dtype=np.float32)
if len(points) < 4:
    st.caption("คลิกบนรูปภาพให้ครบ 4 จุด หรือใช้ปุ่มค้นหาอัตโนมัติ")
    st.stop()
ordered_points = order_points(points)
is_valid, validation_message = validate_quadrilateral(ordered_points)
if not is_valid:
    st.error(validation_message)
    st.stop()
try:
    warped = warp_plate(original_bgr, ordered_points)
    enhanced = enhance_plate(warped, st.session_state["clahe"], st.session_state["unsharp"])
except cv2.error as error:
    st.error(f"ไม่สามารถปรับมุมภาพได้: {error}")
    st.stop()

annotated = draw_points(original_bgr, ordered_points)
left, right = st.columns(2)
with left:
    st.subheader("ภาพต้นฉบับและจุดที่เลือก")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
with right:
    st.subheader("ภาพป้ายที่ปรับมุมแล้ว")
    st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), width="stretch")
    st.download_button("📥 ดาวน์โหลดภาพ PNG", data=image_as_png(enhanced), file_name="license_plate_deskewed.png", mime="image/png", width="stretch")
