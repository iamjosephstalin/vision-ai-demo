import streamlit as st
import os
import json
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image, ImageEnhance, ImageOps
from google.cloud import vision_v1
from google.oauth2 import service_account
import io
import pandas as pd
import cv2
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def resize_if_large(pil_img: Image.Image, max_dimension: int = 1600) -> Image.Image:
    """Resize image if larger than max_dimension to speed up processing."""
    w, h = pil_img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return pil_img

def preprocess_image(pil_img: Image.Image, resize: bool = True) -> Image.Image:
    """Prepares image for better OCR: handle rotation, contrast, sharpness."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    
    # Resize large images to speed up processing
    if resize:
        img = resize_if_large(img)
    
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
    Advanced preprocessing for embossed/raised text on metal castings.
    Uses multi-scale edge detection, bilateral filtering, and morphological operations
    to enhance 3D shadows and edges characteristic of embossed text.
    """
    # Convert PIL to CV2 (RGB -> BGR)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral Filter - Reduces noise while preserving edges (critical for embossed text)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    # 4. Multi-scale edge detection for 3D embossing shadows
    # Sobel edges to detect light/shadow boundaries
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    
    # 5. Adaptive Thresholding - two scales for different text sizes
    # Fine scale for small casting marks
    thresh1 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 3
    )
    # Coarse scale for larger embossed text
    thresh2 = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 25, 7
    )
    
    # 6. Combine edge info with thresholding
    combined = cv2.bitwise_or(thresh1, thresh2)
    combined = cv2.bitwise_or(combined, sobel)
    
    # 7. Morphological operations to clean and enhance text
    # Close small gaps in characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 8. Dilate slightly to make text more solid
    kernel_dilate = np.ones((1,1), np.uint8)
    final = cv2.dilate(clean, kernel_dilate, iterations=1)
    
    # Convert back to PIL
    return Image.fromarray(final)

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
    """Identifies casting markings vs etched plate text with enhanced detection."""
    s = line.strip()
    if not s: return False
    up = s.upper()
    
    # Exclude obvious plate labels
    plate_starters = ("SN", "S/N", "SERIAL", "MODEL", "DN", "PN", "PT", 
                     "BODY", "DISC", "SEAT", "DATE", "WWW.", "MFG", "MFR")
    if up.startswith(plate_starters): return False
    
    # Common casting materials and codes
    casting_materials = ("CF8M", "CF8", "CF3M", "CF3", "WCB", "WCC", "LCB", "LCC",
                        "A351", "A216", "A105", "A182", "316SS", "304SS", "SS316", "SS304")
    if any(mat in up.replace("-", "").replace(" ", "") for mat in casting_materials):
        return True
    
    # Valve/casting type codes (often embossed)
    casting_codes = ("TTV", "DBB", "FLG", "SW", "BW", "THD", "NPT")
    if up in casting_codes:
        return True
    
    # Short alphanumeric codes without spaces (typical casting marks)
    if " " not in up and 2 <= len(up) <= 10:
        # Check if it's alphanumeric (common for casting marks)
        if any(c.isalpha() for c in up) and any(c.isdigit() for c in up):
            return True
        # Pure material codes
        if up.replace("-", "").isalpha() and 2 <= len(up) <= 8:
            return True
    
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
    """Intelligently maps casting marks to appropriate fields with source tracking."""
    up_lines = [c.upper() for c in casting_lines]
    
    # Track where data came from for transparency
    if "data_sources" not in parsed:
        parsed["data_sources"] = {}
    
    # DN (Diameter Nominal) from casting
    if not parsed.get("dn"):
        for u, raw in zip(up_lines, casting_lines):
            if u.startswith("DN") and any(c.isdigit() for c in u):
                parsed["dn"] = raw
                parsed["data_sources"]["dn"] = "casting"
                break
    
    # Material codes - commonly embossed on body
    stainless_materials = ["CF8M", "CF8", "CF3M", "CF3", "316SS", "304SS", "SS316", "SS304"]
    carbon_materials = ["WCB", "WCC", "LCB", "LCC", "A216", "A105"]
    all_materials = stainless_materials + carbon_materials
    
    found_material = None
    for u, raw in zip(up_lines, casting_lines):
        clean_u = u.replace("-", "").replace(" ", "")
        for mat in all_materials:
            if mat in clean_u:
                found_material = raw
                break
        if found_material:
            break
    
    # Map material to body/disc/seat if not already set
    if found_material:
        if not parsed.get("body"):
            parsed["body"] = found_material
            parsed["data_sources"]["body"] = "casting"
        if not parsed.get("disc") and found_material in stainless_materials:
            parsed["disc"] = found_material
            parsed["data_sources"]["disc"] = "casting"
        if not parsed.get("seat") and found_material in stainless_materials:
            parsed["seat"] = found_material
            parsed["data_sources"]["seat"] = "casting"
    
    # Valve type codes
    valve_types = {"TTV": "Triple Offset Butterfly", "DBB": "Double Block & Bleed", 
                   "FLG": "Flanged", "BW": "Butt Weld", "SW": "Socket Weld"}
    for code, desc in valve_types.items():
        if code in up_lines and not parsed.get("model"):
            parsed["model"] = code
            parsed["data_sources"]["model"] = "casting"
            break
    
    # Connection codes for PT (Pressure/Temperature rating)
    connection_codes = ["NPT", "THD", "SW", "BW"]
    for u, raw in zip(up_lines, casting_lines):
        if any(code in u for code in connection_codes) and not parsed.get("pt"):
            parsed["pt"] = raw
            parsed["data_sources"]["pt"] = "casting"
            break
    
    return parsed



# ... (rest of parsing logic stays) ...

# ─────────────────────────────────────────────────────────────────────────────
# 2. UI IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []

# ─────────────────────────────────────────────────────────────────────────────
# UI Controls
# ─────────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

with col2:
    quality_mode = st.selectbox(
        "Quality Mode",
        options=["Fast", "Balanced", "Thorough"],
        index=0,  # Default to Fast for maximum speed
        help="Fast: 1 OCR pass (~3-5s/img)\nBalanced: 2 passes (~6-10s/img)\nThorough: 3 passes (~10-20s/img)"
    )
    
    max_workers = st.slider(
        "Concurrent Images",
        min_value=1,
        max_value=5,
        value=5,  # Max parallelization for speed
        help="Process multiple images in parallel (faster but uses more API quota)"
    )

# Show time estimate
if uploaded_files:
    num_files = len(uploaded_files)
    if quality_mode == "Fast":
        time_per_img = 4
    elif quality_mode == "Balanced":
        time_per_img = 8
    else:
        time_per_img = 15
    
    estimated_time = (num_files * time_per_img) / max_workers
    st.info(f"📊 {num_files} image(s) selected | Estimated time: ~{int(estimated_time)}s ({quality_mode} mode, {max_workers} concurrent)")

if st.button("Process Images") and uploaded_files:
    try:
        client = get_vision_client()
        start_time = time.time()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Helper to chunk list into groups of n
        def chunked(seq, n):
            return (seq[i:i + n] for i in range(0, len(seq), n))

        # Determine OCR passes based on quality mode
        def get_ocr_passes(mode: str) -> List[str]:
            if mode == "Fast":
                return ["normal"]
            elif mode == "Balanced":
                return ["normal", "casting"]
            else:  # Thorough
                return ["normal", "high_contrast", "casting"]
        
        ocr_passes = get_ocr_passes(quality_mode)
        
        # Function to process a single image
        def process_single_image(file, file_idx: int, total_files: int) -> Tuple[str, str, List[str], List[str]]:
            """Process one image with selected OCR passes."""
            filename = file.name
            
            # Load Image
            file.seek(0)  # Reset file pointer
            image_bytes = file.read()
            original_img = Image.open(io.BytesIO(image_bytes))
            
            all_text_parts = []
            
            # OCR Pass 1: Normal (always included)
            if "normal" in ocr_passes:
                preprocessed = preprocess_image(original_img, resize=True)
                text1 = ocr_image(client, preprocessed, mode="document")
                if text1:
                    all_text_parts.append(text1)
            
            # OCR Pass 2: High Contrast (only in Thorough mode)
            if "high_contrast" in ocr_passes:
                preprocessed = preprocess_image(original_img, resize=True)
                hc_img = make_high_contrast(preprocessed)
                text2 = ocr_image(client, hc_img, mode="document")
                if text2:
                    all_text_parts.append(text2)

            # OCR Pass 3: Casting Optimization (Balanced and Thorough)
            if "casting" in ocr_passes:
                casting_img = preprocess_for_casting(original_img)
                text3 = ocr_image(client, casting_img, mode="text")
                if text3:
                    all_text_parts.append(text3)
                
            # Combine & Parse
            raw_text = "\n".join(all_text_parts)
            
            # Collect valid lines
            all_lines = normalize_lines(raw_text)
            
            # Classify lines
            c_lines = [ln for ln in all_lines if looks_like_casting(ln)]
            p_lines = [ln for ln in all_lines if not looks_like_casting(ln)]
            
            return filename, raw_text, c_lines, p_lines
        
        # Helper to chunk list into groups of n
        new_results = []
        
        # Process in groups of 3 (same nameplate from different angles)
        groups = list(chunked(uploaded_files, 3))
        
        total_processed = 0
        
        for g_idx, group in enumerate(groups):
            group_texts = []
            group_casting = []
            group_plate_lines = []
            filenames = []
            
            # Process images in this group concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for file in group:
                    future = executor.submit(process_single_image, file, total_processed, len(uploaded_files))
                    futures[future] = file.name
                    total_processed += 1
                
                # Collect results as they complete
                for future in as_completed(futures):
                    filename, raw_text, c_lines, p_lines = future.result()
                    filenames.append(filename)
                    group_texts.append(raw_text)
                    group_casting.extend(c_lines)
                    group_plate_lines.extend(p_lines)
                    
                    # Update progress
                    elapsed = time.time() - start_time
                    progress = total_processed / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    if progress > 0:
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        status_text.text(f"⏱️ Processed {total_processed}/{len(uploaded_files)} | Elapsed: {int(elapsed)}s | Remaining: ~{int(remaining)}s")
            
            # --- Aggregate & Extract for the whole group ---
            parsed_data = extract_fields(group_plate_lines)
            parsed_data = try_fill_from_casting(parsed_data, group_casting)
            
            # Add metadata
            parsed_data["filename"] = ", ".join(filenames)
            parsed_data["raw_text"] = "\n\n--- NEXT IMAGE ---\n\n".join(group_texts)
            
            # Add visual indicators for data sources
            sources = parsed_data.pop("data_sources", {})
            for field, source in sources.items():
                if source == "casting" and field in parsed_data and parsed_data[field]:
                    parsed_data[field] = f"🔨 {parsed_data[field]}"  # Hammer emoji for casting
            
            # Store casting lines for debugging
            parsed_data["casting_marks"] = ", ".join(group_casting) if group_casting else "None detected"
            
            new_results.append(parsed_data)

        
        st.session_state.results = new_results
        
        # Final status
        total_time = time.time() - start_time
        status_text.text("")
        progress_bar.progress(1.0)
        st.success(f"✅ Processing Complete! Processed {len(uploaded_files)} images in {int(total_time)}s ({quality_mode} mode)")
        
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
    
    # Reorder columns to put filename first, casting marks before raw text
    exclude_cols = ["filename", "raw_text", "casting_marks"]
    cols = ["filename"] + [c for c in df.columns if c not in exclude_cols] + ["casting_marks"]
    # Only include casting_marks if it exists
    cols = [c for c in cols if c in df.columns]
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
