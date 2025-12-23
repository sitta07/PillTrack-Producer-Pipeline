import streamlit as st
import yaml, os, cv2, shutil, glob
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import Custom Modules (ต้องมั่นใจว่าไฟล์เหล่านี้อยู่ในโฟลเดอร์เดียวกัน)
from cloud_manager import CloudManager
from db_manager import DBManager
from engine import AIEngine

# ================= 1. SETUP & CONFIGURATION =================
load_dotenv()
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

def load_config():
    """Load configuration from YAML file safely."""
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()

# --- Extract Config Values ---
ARTIFACTS = config.get('artifacts', {})
PATHS = config.get('paths', {})
SETTINGS = config.get('settings', {})

# Local Paths
PKL_PATH = str(ARTIFACTS.get('pack_vec', 'database/pill_fingerprints.pkl'))
JSON_PATH = str(ARTIFACTS.get('drug_list', 'database/drug_list.json'))
MODEL_PATH = str(ARTIFACTS.get('model', 'models/seg_best_process.pt'))
IMG_DB_ROOT = str(PATHS.get('db_images', 'database_images'))

# S3 Staging Path (Dynamic from Config)
STAGING_S3_PATH = str(PATHS.get('staging_s3', 'staging_models'))

# Engine Settings
DINO_SIZE = SETTINGS.get('dino_size', 224)
YOLO_CONF = SETTINGS.get('yolo_conf', 0.5)

# Ensure essential directories exist
os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
os.makedirs(IMG_DB_ROOT, exist_ok=True)

# ================= 2. INITIALIZATION =================
st.set_page_config(page_title="PillTrack Ops Hub", layout="wide")

cloud = CloudManager(S3_BUCKET)
db = DBManager()

# Alert System (for Success Messages)
if "push_success_msg" in st.session_state:
    st.success(st.session_state.push_success_msg, icon="✅")
    del st.session_state.push_success_msg 

@st.cache_resource
def get_engine():
    """Initialize and Cache AI Engine."""
    return AIEngine(MODEL_PATH, DINO_SIZE)

engine = get_engine()
packs_db = db.load_pkl(PKL_PATH)

def get_base_drug_names(db_dict):
    """Extract clean drug names from DB keys."""
    names = set()
    for k in db_dict.keys():
        # Remove suffixes like _box, _blister, _rot...
        base = k.split('_box')[0].split('_blister')[0].split('_pack')[0].split('_rot')[0] 
        names.add(base)
    return sorted(list(names))

current_drugs = get_base_drug_names(packs_db)

# ================= 3. UI SIDEBAR (SYSTEM & CLOUD) =================
st.sidebar.header("☁️ Cloud & Staging Sync")
s3_ok, s3_status = cloud.check_connection()
st.sidebar.write(f"S3 Connection: `{s3_status}`")

# --- [NEW] PULL FROM STAGING ---
st.sidebar.subheader("📥 Pull Data")
if st.sidebar.button("📥 PULL STAGING MODEL", use_container_width=True, help=f"ดึงไฟล์จาก S3: {STAGING_S3_PATH}"):
    if not S3_BUCKET:
        st.sidebar.error("❌ S3_BUCKET_NAME not found in .env")
    else:
        with st.sidebar.status("Pulling from S3 Staging...", expanded=False) as status:
            try:
                sync_map = {
                    f"{STAGING_S3_PATH}/pill_fingerprints.pkl": PKL_PATH,
                    f"{STAGING_S3_PATH}/drug_list.json": JSON_PATH
                }
                for s3_key, local_path in sync_map.items():
                    status.write(f"Downloading: `{s3_key}`")
                    cloud.s3.download_file(str(S3_BUCKET), s3_key, str(local_path))
                
                status.update(label="✅ Pull Complete!", state="complete")
                st.session_state.push_success_msg = f"Sync with Staging ({STAGING_S3_PATH}) Success!"
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Pull failed: {str(e)}")

st.sidebar.divider()
if st.sidebar.button("🔄 FORCE REFRESH APP", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

# ================= 4. DASHBOARD METRICS =================
st.title("💊 PillTrack: MLOps Producer Hub")
m1, m2, m3 = st.columns(3)
m1.metric("Unique Drugs (SKUs)", len(current_drugs))
m2.metric("Total Embeddings", sum([len(v.get('dino', [])) for v in packs_db.values() if isinstance(v, dict)]))
m3.metric("S3 Backend", "Online" if s3_ok else "Offline")

st.divider()

# ================= 5. DATASET MANAGEMENT =================
st.subheader("🛠️ Dataset Operations")

mode = st.radio("Select Action:", ["Add/Update SKU", "❌ Delete / Cleanup"], horizontal=True)

with st.form("dataset_ops_form"):
    if mode == "Add/Update SKU":
        c1, c2 = st.columns(2)
        with c1:
            input_method = st.radio("Source:", ["Existing SKU", "New SKU"], horizontal=True)
            if input_method == "Existing SKU" and current_drugs:
                drug_name_input = st.selectbox("Select Drug:", current_drugs)
            else:
                drug_name_input = st.text_input("New Drug Name (English):", placeholder="e.g. amoxicillin").strip().lower()
        with c2:
            pack_type = st.selectbox("Product Type:", ["Blister", "Box"])
            
        files_in = st.file_uploader("Upload Drug Images:", accept_multiple_files=True, type=['jpg','jpeg','png'])
    
    else: # DELETE MODE
        all_keys = sorted(list(set([k.split('_rot')[0] for k in packs_db.keys()])))
        target_to_delete = st.selectbox("Select Class to Delete:", all_keys)

    submit_btn = st.form_submit_button("🚀 EXECUTE ACTION", type="primary", use_container_width=True)

    if submit_btn:
        # --- LOGIC: DELETE ---
        if mode == "❌ Delete / Cleanup" and target_to_delete:
            keys_to_del = [k for k in packs_db.keys() if k.startswith(target_to_delete)]
            for k in keys_to_del: del packs_db[k]
            
            # Remove Image Folder
            path = os.path.join(IMG_DB_ROOT, target_to_delete)
            if os.path.exists(path): shutil.rmtree(path)
            
            db.save_pkl(packs_db, PKL_PATH)
            db.generate_metadata(get_base_drug_names(packs_db), JSON_PATH)
            st.session_state.push_success_msg = f"Deleted {target_to_delete} successfully."
            st.rerun()

        # --- LOGIC: ADD/UPDATE ---
        elif mode == "Add/Update SKU" and drug_name_input and files_in:
            final_class_name = f"{drug_name_input}_{pack_type.lower()}"
            save_dir = os.path.join(IMG_DB_ROOT, final_class_name)
            
            # Clean old data for this class
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            for k in [k for k in packs_db.keys() if k.startswith(final_class_name)]: del packs_db[k]
            
            with st.status(f"Processing {final_class_name}...") as status:
                for i, item in enumerate(files_in):
                    # Read image
                    file_bytes = np.frombuffer(item.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    if img is None: continue
                    
                    # Save raw image for DB
                    cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), img)
                    
                    # AI Processing: Detect & Crop
                    crop = engine.detect_and_crop(img, YOLO_CONF)
                    if crop is None: continue
                    
                    # Feature Extraction with 4 Rotations
                    for angle in [0, 90, 180, 270]:
                        if angle == 0: rot = crop
                        elif angle == 90: rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 180: rot = cv2.rotate(crop, cv2.ROTATE_180)
                        elif angle == 270: rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                        dino_vec, _ = engine.extract_features(rot)
                        db.insert_data(packs_db, f"{final_class_name}_rot{angle}", dino_vec, None)
                
                # Persistence
                db.save_pkl(packs_db, PKL_PATH)
                db.generate_metadata(get_base_drug_names(packs_db), JSON_PATH)
                status.update(label=f"✅ {final_class_name} Updated!", state="complete")
            
            st.session_state.push_success_msg = f"Successfully processed {len(files_in)} images for {final_class_name}"
            st.rerun()

# ================= 6. PUSH TO STAGING (OVERWRITE) =================
st.divider()
st.subheader("📤 Cloud Deployment")
st.info(f"Push current local artifacts to S3 Staging: `{STAGING_S3_PATH}/`")

if st.button("🚀 PUSH TO STAGING (OVERWRITE S3)", type="primary", use_container_width=True):
    with st.status("Uploading to S3 Staging...") as status:
        try:
            # Sync Local -> S3 (Overwrite)
            cloud.upload_file(PKL_PATH, f"{STAGING_S3_PATH}/pill_fingerprints.pkl")
            cloud.upload_file(JSON_PATH, f"{STAGING_S3_PATH}/drug_list.json")
            
            status.update(label="🚀 Staging Updated on Cloud!", state="complete")
            st.toast("Artifacts pushed successfully!")
        except Exception as e:
            st.error(f"Push failed: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")