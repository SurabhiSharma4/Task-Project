import streamlit as st
import cv2
import numpy as np
from rembg import remove
from ultralytics import YOLO
import tempfile
import os
import time

model = YOLO("yolov8x.pt")

def process_alpha(image, threshold=80):
    """
    Thresholds alpha channel to remove partial transparency:
    alpha > threshold -> 255, else -> 0.
    """
    if image.shape[-1] == 4:
        alpha = image[:, :, 3]
        _, alpha_thresh = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)
        image[:, :, 3] = alpha_thresh
    return image

def parse_surface_labels(prompt):
    """
    Map user prompt keywords to YOLO labels for surfaces (table, nightstand, desk, etc.).
    """
    label_map = {
    "table": ["dining table", "table", "coffee table", "work table", "side table"],
    "desk": ["desk", "counter", "work desk", "office desk", "study desk"],
    "shelf": ["shelf", "bookshelf", "wall shelf", "storage shelf"],
    "nightstand": ["nightstand", "bedside table", "bedside stand"],
    "bedside": ["nightstand", "bedside table", "bedside stand"],
    "couch": ["couch", "sofa", "loveseat", "sectional sofa"],
    "sofa": ["couch", "sofa", "loveseat", "recliner"],
    "chair": ["chair", "armchair", "dining chair", "office chair"],
    "bench": ["bench", "park bench", "indoor bench", "storage bench"],
    "cabinet": ["cabinet", "kitchen cabinet", "storage cabinet", "tv cabinet"],
    "dresser": ["dresser", "chest of drawers", "wardrobe"],
    "tv stand": ["tv stand", "entertainment center", "media console"],
    "floor": ["floor", "ground", "wooden floor", "carpeted floor", "tile floor"],
    "bed": ["bed", "single bed", "double bed", "king-size bed", "queen bed"],
    "stool": ["stool", "bar stool", "footrest", "ottoman"],
    "countertop": ["countertop", "kitchen counter", "bathroom counter"],
    "window sill": ["window sill", "window ledge"],
    "fireplace": ["fireplace", "mantel", "fireplace shelf"],
    "washing machine": ["washing machine", "laundry machine", "dryer"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "sink": ["sink", "kitchen sink", "bathroom sink"],
    "bathtub": ["bathtub", "tub", "jacuzzi"]
}

    prompt = prompt.lower()
    matched_labels = []
    for kw, labels in label_map.items():
        if any(word in prompt for word in kw.split()):
            matched_labels.extend(labels)
    if not matched_labels:
        matched_labels = sum(label_map.values(), [])
    return matched_labels

def detect_best_surface(image_path, valid_labels, conf_thresh=0.5):
    """
    Runs YOLO detection on the lifestyle image. Returns bounding box (x1,y1,x2,y2) or None.
    """
    results = model.predict(image_path, conf=conf_thresh)
    best_area = 0
    best_box = None
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id].lower()
            conf = float(box.conf[0])
            if label in valid_labels and conf >= conf_thresh:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
    return best_box

def compute_scale_factor(product_h, surface_h):
    """
    Heuristic scaling so product isn't too large or too small relative to surface height.
    """
    scale_factor = surface_h / (2.5 * product_h)
    return max(0.3, min(scale_factor, 1.0))

def place_object_within_box(box, product_h, product_w, prompt):
    """
    Place product inside bounding box (top/bottom/left/right).
    Returns (x_center, y_center).
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w // 2
    y_center = y1 + box_h // 2

    p = prompt.lower()
    if "top" in p:
        y_center = y1 + product_h // 2 + int(box_h * 0.05)
    elif "bottom" in p:
        y_center = y2 - product_h // 2 - int(box_h * 0.05)
    if "left" in p:
        x_center = x1 + product_w // 2 + int(box_w * 0.05)
    elif "right" in p:
        x_center = x2 - product_w // 2 - int(box_w * 0.05)

    return x_center, y_center

def blend_product(
    lifestyle_img, product_img, x_center, y_center
):
    """
    Blends product_img onto lifestyle_img at center (x_center, y_center).
    Returns a new composite image (OpenCV BGR).
    """
    composite = lifestyle_img.copy()
    ph, pw = product_img.shape[:2]
    yc = y_center - ph // 2
    xc = x_center - pw // 2
    yc2 = yc + ph
    xc2 = xc + pw

    # Clamp to boundaries
    yc = max(0, yc); yc2 = min(composite.shape[0], yc2)
    xc = max(0, xc); xc2 = min(composite.shape[1], xc2)
    if yc2 <= yc or xc2 <= xc:
        # Product is out of frame
        return composite

    final_w = xc2 - xc
    final_h = yc2 - yc
    # Possibly resize product if region is smaller
    product_region = product_img
    if final_w < pw or final_h < ph:
        product_region = cv2.resize(product_region, (final_w, final_h), interpolation=cv2.INTER_AREA)
        ph, pw = final_h, final_w

    # Extract mask
    if product_region.shape[-1] == 4:
        mask = product_region[:, :, 3]
    else:
        mask = np.ones((ph, pw), dtype=np.uint8) * 255

    for c in range(3):
        composite[yc:yc+ph, xc:xc+pw, c] = (
            product_region[:ph, :pw, c] * (mask[:ph, :pw] / 255.0) +
            composite[yc:yc+ph, xc:xc+pw, c] * (1.0 - mask[:ph, :pw] / 255.0)
        )

    return composite

#########################
# STREAMLIT APP
#########################

st.title("AI Product Placement with Manual Adjustment")

# Step A: User uploads images
product_file = st.file_uploader("Product Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
lifestyle_file = st.file_uploader("Lifestyle Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
prompt = st.text_area("Prompt (e.g., 'Place the lamp on the bedside table')")

if st.button("Initial AI Placement"):
    if not product_file or not lifestyle_file:
        st.error("Please upload both images.")
        st.stop()

    # 1) Read images
    product_cv2 = cv2.imdecode(np.frombuffer(product_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    lifestyle_cv2 = cv2.imdecode(np.frombuffer(lifestyle_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if product_cv2 is None or lifestyle_cv2 is None:
        st.error("Could not read the images.")
        st.stop()

    # 2) Remove background
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(product_file.getbuffer())
        product_path = temp_file.name

    with open(product_path, "rb") as f:
        product_bytes = f.read()
    bg_removed = remove(product_bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_out:
        temp_out.write(bg_removed)
        no_bg_path = temp_out.name

    product_no_bg = cv2.imread(no_bg_path, cv2.IMREAD_UNCHANGED)
    if product_no_bg is None:
        st.error("Background removal failed.")
        st.stop()

    product_no_bg = process_alpha(product_no_bg, threshold=80)

    # 3) YOLO detection
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_lifestyle:
        temp_lifestyle.write(lifestyle_file.getbuffer())
        lifestyle_path = temp_lifestyle.name

    # get relevant labels from prompt
    valid_labels = parse_surface_labels(prompt)
    best_box = detect_best_surface(lifestyle_path, valid_labels, conf_thresh=0.5)

    ph, pw = product_no_bg.shape[:2]

    # 4) Automatic scale + placement
    if best_box:
        x1, y1, x2, y2 = best_box
        surface_h = (y2 - y1)
        scale_factor = compute_scale_factor(ph, surface_h)
        new_w = int(pw * scale_factor)
        new_h = int(ph * scale_factor)
        product_resized = cv2.resize(product_no_bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_center, y_center = place_object_within_box(best_box, new_h, new_w, prompt)
    else:
        # fallback center
        st.info("...")
        min_dim = min(lifestyle_cv2.shape[0], lifestyle_cv2.shape[1])
        scale_factor = (0.3 * min_dim) / ph
        scale_factor = max(0.2, min(scale_factor, 1.0))
        new_w = int(pw * scale_factor)
        new_h = int(ph * scale_factor)
        product_resized = cv2.resize(product_no_bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_center = lifestyle_cv2.shape[1] // 2
        y_center = lifestyle_cv2.shape[0] // 2

    # 5) Blend
    composite = blend_product(lifestyle_cv2, product_resized, x_center, y_center)

    # 6) Show + store in session_state
    st.session_state["product_resized"] = product_resized
    st.session_state["lifestyle_cv2"] = lifestyle_cv2
    st.session_state["composite"] = composite
    st.session_state["x_center"] = x_center
    st.session_state["y_center"] = y_center

    st.image(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB), caption="Initial AI Composite")

# Step B: Manual Adjustments
if "composite" in st.session_state:
    st.subheader("Manual Adjustments")
    offset_x = st.slider("Horizontal Offset", -500, 500, 0)
    offset_y = st.slider("Vertical Offset", -800, 800, 0)
    manual_scale = st.slider("Additional Scale Factor", 0.1, 3.0, 1.0)

    if st.button("Apply Manual Adjustments"):
        product_resized = st.session_state["product_resized"]
        lifestyle_cv2 = st.session_state["lifestyle_cv2"].copy()

        # Re-scale product by manual_scale
        ph, pw = product_resized.shape[:2]
        new_w = int(pw * manual_scale)
        new_h = int(ph * manual_scale)
        product_resized2 = cv2.resize(product_resized, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Re-blend with offsets
        x_center = st.session_state["x_center"] + offset_x
        y_center = st.session_state["y_center"] + offset_y

        composite2 = blend_product(lifestyle_cv2, product_resized2, x_center, y_center)

        st.image(cv2.cvtColor(composite2, cv2.COLOR_BGR2RGB), caption="Manually Adjusted Composite")

        # Save the file directly in the working directory
        out_path = "manual_composite.png"

        # Save the composite image inside the folder
        cv2.imwrite(out_path, composite2)

        # Provide the download button
        with open(out_path, "rb") as file:
            st.download_button("Download Manually Adjusted Composite", file, file_name="manual_composite.png", mime="image/png")