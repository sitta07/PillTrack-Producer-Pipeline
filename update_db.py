import streamlit as st
import os
import cv2
import torch
import numpy as np
import pickle
import shutil
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

## streamlit run ./update_db.py ##

# =================  CONFIGURATION =================
# ตั้งค่า Path ให้ตรงกับเครื่อง Local (Pi/Mac)
BASE_DIR = os.getcwd()
DB_IMAGES_FOLDER = os.path.join(BASE_DIR, 'database_images')
DB_PKL_FOLDER = os.path.join(BASE_DIR)

# ไฟล์ Database เป้าหมาย
FILES = {
    'pill_vec': os.path.join(DB_PKL_FOLDER, 'db_pills_dino.pkl'),
    'pack_vec': os.path.join(DB_PKL_FOLDER, 'db_packs_dino.pkl'),
    'pill_col': os.path.join(DB_PKL_FOLDER, 'colors_pills.pkl'),
    'pack_col': os.path.join(DB_PKL_FOLDER, 'colors_packs.pkl')
}

MODEL_PATH = 'seg_best_process.pt' # โมเดล YOLO ของคุณ
DINO_SIZE = 448 # ต้องตรงกับ Main 448

# ================= 🔧 SYSTEM SETUP =================
st.set_page_config(page_title="PillTrack DB Manager", layout="wide", page_icon="💊")

# ใช้ Cache เพื่อโหลด Model แค่ครั้งเดียว (เร็วขึ้นมาก)
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino.eval().to(device)
    
    # Load YOLO
    try:
        yolo = YOLO(MODEL_PATH)
    except:
        yolo = YOLO('yolov8n-seg.pt') # Fallback
        
    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((DINO_SIZE, DINO_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return dino, yolo, preprocess, device

# โหลดของรอไว้เลย
dino_model, yolo_model, preprocess_pipeline, device = load_models()

# =================  HELPER FUNCTIONS =================
def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f: return pickle.load(f)
    return {}

def save_pkl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(data, f)

def get_vector(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    t = preprocess_pipeline(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = dino_model(t)
        if isinstance(output, dict): output = output['x_norm_clstoken']
        vec = output.flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

def get_smart_color(img_bgr):
    try:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.array([0, 40, 40]), np.array([180, 255, 230]))
        pixels = img_hsv[mask > 0]
        if len(pixels) < 50: pixels = img_hsv.reshape(-1, 3)
        return np.mean(pixels, axis=0)
    except: return np.zeros(3)

# ================= 🚀 CORE LOGIC: PROCESS DRUG =================
def process_drug_update(name, drug_type, uploaded_files, status_box, progress_bar):
    # 1. Create Folder & Save Images
    folder_name = f"{name}_{drug_type}"
    save_path = os.path.join(DB_IMAGES_FOLDER, folder_name)
    
    # ถ้ามีโฟลเดอร์อยู่แล้ว ลบสร้างใหม่ (Clean Start)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    status_box.info(f"📂 Created folder: {folder_name}")
    
    saved_images = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_images.append(file_path)
    
    status_box.success(f"✅ Saved {len(saved_images)} images.")
    
    # 2. Prepare Database
    is_pill = (drug_type == 'pill')
    vec_key = 'pill_vec' if is_pill else 'pack_vec'
    col_key = 'pill_col' if is_pill else 'pack_col'
    
    db_vec = load_pkl(FILES[vec_key])
    db_col = load_pkl(FILES[col_key])

    # Clean old keys
    keys_to_del = [k for k in db_vec.keys() if k.startswith(folder_name)]
    for k in keys_to_del: del db_vec[k]

    # 3. Processing Loop (DINO + YOLO)
    temp_colors = []
    total_imgs = len(saved_images)
    
    for i, img_path in enumerate(saved_images):
        progress = (i + 1) / total_imgs
        progress_bar.progress(progress, text=f"Processing image {i+1}/{total_imgs}...")
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # YOLO Crop
        results = yolo_model(img, verbose=False, conf=0.6)
        if len(results[0].boxes) > 0:
            box = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h, w = img.shape[:2]
            pad = 30
            cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
            cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
            crop = img[cy1:cy2, cx1:cx2]
        else:
            crop = img # Fallback
            
        # 4 Rotations
        rotations = [(0, "_rot0"), (90, "_rot90"), (180, "_rot180"), (270, "_rot270")]
        for angle, suffix in rotations:
            rot_img = crop.copy()
            if angle == 90: rot_img = cv2.rotate(rot_img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180: rot_img = cv2.rotate(rot_img, cv2.ROTATE_180)
            elif angle == 270: rot_img = cv2.rotate(rot_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Get Vector
            vec = get_vector(rot_img)
            full_key = f"{folder_name}{suffix}"
            
            # Save as list [vec] to match format
            if full_key not in db_vec: db_vec[full_key] = []
            db_vec[full_key].append(vec)
            
            if angle == 0 and is_pill:
                h_c, w_c = rot_img.shape[:2]
                center = rot_img[h_c//4:h_c*3//4, w_c//4:w_c*3//4]
                if center.size > 0: temp_colors.append(get_smart_color(center))

    # 4. Save Changes
    if temp_colors:
        db_col[folder_name] = np.mean(temp_colors, axis=0)
        
    save_pkl(db_vec, FILES[vec_key])
    save_pkl(db_col, FILES[col_key])
    
    return True

# ================= 🖥️ UI LAYOUT =================

# --- SIDEBAR: DB MONITOR ---
st.sidebar.title("Database Monitor")
st.sidebar.markdown("---")

# Load Stats
try:
    pills = load_pkl(FILES['pill_vec'])
    packs = load_pkl(FILES['pack_vec'])
    
    # Extract unique drug names
    pill_names = sorted(list(set([k.split('_rot')[0] for k in pills.keys()])))
    pack_names = sorted(list(set([k.split('_rot')[0] for k in packs.keys()])))
    
    st.sidebar.subheader(f"🟢 Pills ({len(pill_names)})")
    st.sidebar.dataframe([n.replace('_pill','') for n in pill_names], use_container_width=True, hide_index=True)
    
    st.sidebar.subheader(f"📦 Packs ({len(pack_names)})")
    st.sidebar.dataframe([n.replace('_pack','') for n in pack_names], use_container_width=True, hide_index=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Total Vectors: {len(pills) + len(packs)}")
    
except Exception as e:
    st.sidebar.error(f"Cannot load DB: {e}")


# --- MAIN PAGE: ADD/UPDATE ---
st.title("🛠️ PillTrack DB Manager")
st.markdown("Add new drugs or update existing ones directly from here.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Drug Info")
    drug_name = st.text_input("Drug Name (English)", placeholder="e.g. paracetamol").strip().lower()
    drug_type = st.radio("Type", ["pill", "pack"], horizontal=True)
    
    st.subheader("2. Upload Images")
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} images.")
        # Preview first 3 images
        st.image([f for f in uploaded_files][:3], width=100, caption=[f.name for f in uploaded_files][:3])

with col2:
    st.subheader("3. Actions")
    status_box = st.empty()
    progress_bar = st.empty()
    
    build_btn = st.button("UPDATE DATABASE", type="primary", use_container_width=True)
    
    if build_btn:
        if not drug_name:
            st.error("Please enter a drug name!")
        elif not uploaded_files:
            st.error("Please upload at least one image!")
        else:
            status_box.info("🚀 Starting process... Please wait.")
            try:
                success = process_drug_update(drug_name, drug_type, uploaded_files, status_box, progress_bar)
                if success:
                    st.balloons()
                    status_box.success(f"🎉 Successfully updated: {drug_name.upper()}")
                    st.success("Database file (.pkl) has been updated! Please restart main.py to see changes.")
            except Exception as e:
                status_box.error(f"❌ Error: {e}")
                st.exception(e)

st.markdown("---")
st.caption("PillTrack System | DINOv2 + YOLO Engine | Running on Streamlit")