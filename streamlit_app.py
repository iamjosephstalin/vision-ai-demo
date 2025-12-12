import streamlit as st
import os
import json
import logging
from typing import List, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageOps
from google.cloud import vision_v1
from google.oauth2 import service_account
import io
import pandas as pd
import cv2
import numpy as np
from dotenv import load_dotenv

# Load env vars from .env if present (for local dev)
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-ocr")

st.set_page_config(page_title="NDT Nameplate OCR", layout="wide")

st.title("🔩 NDT Nameplate OCR Tool")
st.markdown("Upload a valve nameplate image to extract technical specifications.")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE LOGIC (Ported from backend/main.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_vision_client():
    """
    Creates a Google Vision Client.
    Prioritizes:
    1. Streamlit Secrets (st.secrets["gcp_service_account"])
    2. Environment Variable JSON (GOOGLE_APPLICATION_CREDENTIALS_JSON)
    3. Environment Variable File Path (GOOGLE_APPLICATION_CREDENTIALS)
    4. Local hardcoded fallback (for zero-config dev)
    """
    # 1. Try Streamlit Secrets (Best for Streamlit Cloud)
    try:
        if "gcp_service_account" in st.secrets:
            try:
                info = dict(st.secrets["gcp_service_account"])
                creds = service_account.Credentials.from_service_account_info(info)
                return vision_v1.ImageAnnotatorClient(credentials=creds)
            except Exception as e:
                logger.error(f"Failed to load generic st.secrets: {e}")
    except Exception:
        pass # No secrets.toml found, continue


    # 2. Try Env Var with JSON content (Best for Hugging Face)
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json and creds_json.strip().startswith("{"):
        try:
            info = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(info)
            return vision_v1.ImageAnnotatorClient(credentials=creds)
        except Exception as e:
            logger.error(f"Failed to load from GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")

    # 3. Fallback: Check for specific local key file if env var is missing
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        # Auto-detect known key file or generic secrets.json in root
        for key_file in ["secrets.json", "image-extract-476710-c6a143e5254f.json"]:
            if os.path.exists(key_file):
                logger.info(f"Using local key file: {key_file}")
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_file)
                break


    # 4. Fallback to default/file based (Local Dev)
    return vision_v1.ImageAnnotatorClient()


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Prepares image for better OCR: handle rotation, contrast, sharpness."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = img.size
    pad = int(min(w, h) * 0.01)
    if pad > 0:
        img = img.crop((pad, pad, w - pad, h - pad))
    
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img

def make_high_contrast(img: Image.Image) -> Image.Image:
    """Creates a high-contrast inverted version for difficult text."""
    g = img.convert("L")
    g = ImageEnhance.Contrast(g).enhance(3.0)
    g = ImageEnhance.Sharpness(g).enhance(2.5)
    g = ImageOps.invert(g)
    w, h = g.size
    g = g.resize((w * 2, h * 2))
    return g

def preprocess_for_casting(pil_img: Image.Image) -> Image.Image:
    """
    Optimized for embossed metal letters (casting marks).
    Uses CLAHE + Adaptive Thresholding to find 3D edges.
    """
    # Convert PIL to CV2 (RGB -> BGR)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Adaptive Thresholding (Gaussian)
    # This is key for casting: compares pixel to neighbors (local contrast)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 19, 5  # blockSize 19, C 5
    )
    
    # 4. Denoise slightly (Morphological Open)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Convert back to PIL
    return Image.fromarray(clean)

def ocr_image(client, pil_img: Image.Image, mode="document") -> str:
    """Runs Vision API on the image."""
    # Convert PIL to bytes
    with io.BytesIO() as output:
        pil_img.save(output, format="JPEG", quality=95)
        content = output.getvalue()

    image = vision_v1.types.Image(content=content)
    
    if mode == "document":
        resp = client.document_text_detection(image=image)
    else:
        resp = client.text_detection(image=image)
        
    if resp.error.message:
        raise Exception(resp.error.message)
        
    if mode == "document" and resp.full_text_annotation:
        return resp.full_text_annotation.text or ""
    elif mode != "document" and resp.text_annotations:
        return resp.text_annotations[0].description
    
    return ""

def is_valid_line(line: str) -> bool:
    """Filters out likely noise."""
    s = line.strip()
    if not s: return False
    # Relaxed length check
    if len(s) > 60 and " " not in s: return False
    allowed_symbols = " .-/:()\"'°+," # Added +, and °
    bad_count = sum(1 for c in s if not c.isalnum() and c not in allowed_symbols)
    # Relaxed ratio to 0.5 (50% symbols allowed)
    if len(s) > 5 and (bad_count / len(s)) > 0.5: return False
    return True

def normalize_lines(txt: str) -> List[str]:
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return [ln for ln in lines if is_valid_line(ln)]

def looks_like_casting(line: str) -> bool:

    """Identifies casting markings vs etched plate text."""
    s = line.strip()
    if not s: return False
    up = s.upper()
    plate_starters = ("SN", "S/N", "MODEL", "DN", "PN", "PT", "BODY", "DISC", "SEAT", "DATE", "WWW.")
    if up.startswith(plate_starters): return False
    if " " not in up and 2 <= len(up) <= 8: return True
    return False

def extract_fields(lines: List[str]) -> dict:
    """Parses raw lines into structured fields."""
    data = {
        "serial_number": None, "model": None, "dn": None, "pn": None,
        "pt": None, "body": None, "disc": None, "seat": None,
        "temp": None, "date": None
    }
    
    for ln in lines:
        up = ln.upper().strip()
        
        if not data["serial_number"] and ("SN " in up or up.startswith("SN") or "S/N" in up):
            data["serial_number"] = ln
            continue
        if not data["model"] and up.startswith("MODEL"):
            parts = ln.split(None, 1)
            data["model"] = parts[1] if len(parts) > 1 else ln
            continue
        if not data["dn"] and up.startswith("DN"):
            data["dn"] = ln
            continue
        if not data["pn"] and up.startswith("PN"):
            data["pn"] = ln
            continue
        if not data["pt"] and up.startswith("PT"):
            data["pt"] = ln
            continue
        if not data["body"] and "BODY" in up:
            data["body"] = ln
            continue
        if not data["disc"] and "DISC" in up:
            data["disc"] = ln
            continue
        if not data["seat"] and "SEAT" in up:
            data["seat"] = ln
            continue
        if not data["temp"] and ("T(" in up or "°C" in up or up.startswith("T°")):
            if any(c.isdigit() for c in ln):
                data["temp"] = ln
            continue
        if not data["date"] and up.startswith("DATE"):
            data["date"] = ln.lower().replace("date", "").strip()
            continue
            
    return data

def try_fill_from_casting(parsed: dict, casting_lines: List[str]) -> dict:
    """Heuristics to fill missing fields from casting marks."""
    up_lines = [c.upper() for c in casting_lines]
    
    if not parsed.get("dn"):
        for u in up_lines:
            if u.startswith("DN"):
                parsed["dn"] = u; break
    
    cf8m = None
    for u, raw in zip(up_lines, casting_lines):
        if "CF8M" in u.replace("-", "").replace(" ", ""):
            cf8m = raw; break
            
    if cf8m:
        if not parsed.get("disc"): parsed["disc"] = cf8m
        if not parsed.get("body"): parsed["body"] = cf8m
        
    if not parsed.get("model"):
        if "TTV" in up_lines: parsed["model"] = "TTV"
            
    return parsed



# ... (rest of parsing logic stays) ...

# ─────────────────────────────────────────────────────────────────────────────
# 2. UI IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []

uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if st.button("Process Images") and uploaded_files:
    try:
        client = get_vision_client()
        
        progress_bar = st.progress(0)
        
        # Helper to chunk list into groups of n
        def chunked(seq, n):
            return (seq[i:i + n] for i in range(0, len(seq), n))

        new_results = []
        
        # Process in groups of 3
        groups = list(chunked(uploaded_files, 3))
        
        for g_idx, group in enumerate(groups):
            group_texts = []
            group_casting = []
            group_plate_lines = []
            filenames = []

            # Status update
            st.text(f"Processing Group {g_idx + 1}/{len(groups)} ({len(group)} images)...")

            for file in group:
                filenames.append(file.name)
                # Load Image
                image_bytes = file.read()
                original_img = Image.open(io.BytesIO(image_bytes))
                
                # Preprocess
                preprocessed = preprocess_image(original_img)
                
                # OCR Pass 1 (Normal)
                text1 = ocr_image(client, preprocessed, mode="document")
                
                # OCR Pass 2 (High Contrast)
                hc_img = make_high_contrast(preprocessed)
                text2 = ocr_image(client, hc_img, mode="document")

                # OCR Pass 3 (Casting / Embossed Optimization)
                casting_img = preprocess_for_casting(original_img)
                text3 = ocr_image(client, casting_img, mode="text") # 'text' mode often better for sparse casting words
                
                # Combine & Parse
                raw_text = (text1 or "") + "\n" + (text2 or "") + "\n" + (text3 or "")
                
                # Collect valid lines
                all_lines = normalize_lines(raw_text)
                
                # Classify lines
                c_lines = [ln for ln in all_lines if looks_like_casting(ln)]
                p_lines = [ln for ln in all_lines if not looks_like_casting(ln)]
                
                group_texts.append(raw_text)
                group_casting.extend(c_lines)
                group_plate_lines.extend(p_lines)
            
            # --- Aggregate & Extract for the whole group ---
            parsed_data = extract_fields(group_plate_lines)
            parsed_data = try_fill_from_casting(parsed_data, group_casting)
            
            # Add metadata
            parsed_data["filename"] = ", ".join(filenames)
            parsed_data["raw_text"] = "\n\n--- NEXT IMAGE ---\n\n".join(group_texts)
            
            new_results.append(parsed_data)
            
            # Update progress bar
            progress_bar.progress((g_idx + 1) / len(groups))

            
        st.session_state.results = new_results
        st.success("Processing Complete!")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure you have set up Google Cloud Credentials correctly (either via .env or Secrets).")

# Display Results
if st.session_state.results:
    st.divider()
    st.subheader("Extracted Data (Editable)")
    
    # Create DataFrame
    df = pd.DataFrame(st.session_state.results)
    
    # Clean up display: Replace None with empty string
    df = df.fillna("")
    
    # Reorder columns to put filename first
    exclude_cols = ["filename", "raw_text"]
    cols = ["filename"] + [c for c in df.columns if c not in exclude_cols]
    df = df[cols]
    
    # Data Editor
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    
    # Download Button
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ocr_results.csv",
        mime="text/csv",
    )
    
    # Debug: Show Raw details for selected row?
    st.divider()
    with st.expander("Debugging: Raw OCR Output for Inspection"):
        selected_file = st.selectbox("Select file to view raw text:", [r["filename"] for r in st.session_state.results])
        if selected_file:
            raw = next((r["raw_text"] for r in st.session_state.results if r["filename"] == selected_file), "")
            st.text_area("Raw Text Content", raw, height=300)
