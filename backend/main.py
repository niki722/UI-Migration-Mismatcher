 
# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pathlib import Path
from typing import List
import shutil
import json
import glob
import asyncio
from utils import load_latest_results
from difflib import SequenceMatcher
from reportlab.lib import colors
# Import specialized agents
from agent import specialized_graph  
import easyocr
from ultralytics import YOLO
import cv2
from skimage.metrics import structural_similarity as ssim
# ---------------------------
# Setup
# ---------------------------
app = FastAPI(title="FeatureMirror - UI Analysis via Agents")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
 
# Suppress asyncio CancelledError noise
asyncio.get_event_loop().set_exception_handler(
    lambda loop, context: None if isinstance(context.get("exception"), asyncio.CancelledError)
    else loop.default_exception_handler(context)
)
 
# ---------------------------
# Upload folders
# ---------------------------
UPLOAD_DIR = Path("uploads")
OLD_DIR = UPLOAD_DIR / "old"
NEW_DIR = UPLOAD_DIR / "new"
OLD_DIR.mkdir(parents=True, exist_ok=True)
NEW_DIR.mkdir(parents=True, exist_ok=True)
 
ocr_reader = easyocr.Reader(['en'], gpu=False)
yolo_model = YOLO("yolov8n.pt")  # Replace with your trained UI detection model
 
# ---------------------------
# Helper functions
# ---------------------------
def save_files(files: List[UploadFile], folder: Path) -> List[Path]:
    paths = []
    for f in files:
        path = folder / f.filename
        with open(path, "wb") as file_obj:
            shutil.copyfileobj(f.file, file_obj)
        paths.append(path)
    return paths
 
def pair_files(old_files: List[Path], new_files: List[Path]):
    """
    Only pair files with exactly the same stem.
    Returns list of tuples (old_file, new_file)
    """
    new_lookup = {f.stem: f for f in new_files}
    pairs = []
 
    for old in old_files:
        if old.stem in new_lookup:
            pairs.append((old, new_lookup[old.stem]))
 
    return pairs
 
def compare_regions(img1, img2, bbox1, bbox2):
    """
    Compare cropped regions from old and new screenshots.
    bbox: [x_min, y_min, x_max, y_max] (YOLO format or OCR bbox converted)
    Returns SSIM similarity score.
    """
    # Extract regions
    x1, y1, x2, y2 = map(int, bbox1)
    region1 = img1[y1:y2, x1:x2]
    x1, y1, x2, y2 = map(int, bbox2)
    region2 = img2[y1:y2, x1:x2]
 
    if region1.size == 0 or region2.size == 0:
        return 0.0
 
    # Resize to match shapes
    region2 = cv2.resize(region2, (region1.shape[1], region1.shape[0]))
 
    # Convert to grayscale
    region1_gray = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
    region2_gray = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
 
    # Compute SSIM
    score, _ = ssim(region1_gray, region2_gray, full=True)
    return score
 
def run_yolo(image_path: Path, conf_thresh: float = 0.3):
    """Detect UI elements with YOLO and filter by confidence."""
    results = yolo_model(str(image_path))
    detections = []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            conf = float(conf)
            if conf >= conf_thresh:
                detections.append({
                    "class_id": int(cls),
                    "confidence": conf,
                    "xyxy": box.tolist(),
                    "class_name": yolo_model.names[int(cls)] if hasattr(yolo_model, 'names') else str(cls)
                })
    return detections
 
def run_ocr(image_path: Path, conf_thresh: float = 0.5):
    """Extract text with OCR and filter low-confidence results."""
    img = cv2.imread(str(image_path))
    ocr_results = ocr_reader.readtext(img)
    processed = []
    for bbox, text, conf in ocr_results:
        if conf >= conf_thresh:
            processed.append({
                "text": text.strip(),
                "confidence": float(conf),
                "bbox": [float(coord) for point in bbox for coord in point]
 
            })
    return processed
def format_pixel_warning(diff):
    label = diff.get("text") or diff.get("element") or "Unknown"
    ssim_score = diff.get("ssim_score", 0)
    status = diff.get("status", "")
 
    # Describe severity
    if ssim_score < 0.3:
        severity = "Major change"
    elif ssim_score < 0.6:
        severity = "Moderate change"
    else:
        severity = "Minor shift"
 
    # Human-readable message
    return f"{label}: {status.replace('/', ' ').title()} ({severity}, similarity: {ssim_score:.2f})"
 
def analyze_pair(old_file: Path, new_file: Path):
    old_img = cv2.imread(str(old_file))
    new_img = cv2.imread(str(new_file))
 
    old_yolo = run_yolo(old_file)
    new_yolo = run_yolo(new_file)
    old_ocr = run_ocr(old_file)
    new_ocr = run_ocr(new_file)
 
    pixel_differences = []
 
    # --- Compare YOLO elements (bounding boxes) ---
    for e_old in old_yolo:
        elem = e_old.get("class_name")
        old_box = e_old.get("xyxy")
        match = next((e for e in new_yolo if e.get("class_name") == elem), None)
        if match:
            ssim_score = compare_regions(old_img, new_img, old_box, match["xyxy"])
            if ssim_score < 0.8:  # threshold for UI displacement/visual change
                pixel_differences.append({
                    "element": elem,
                    "old_box": old_box,
                    "new_box": match["xyxy"],
                    "ssim_score": float(ssim_score),
                    "status": "SHIFTED / CHANGED"
                })
 
    # --- Compare OCR texts (regions) ---
    for o_item in old_ocr:
        old_text = o_item["text"]
        old_bbox = o_item["bbox"]
        old_box = [min(old_bbox[0::2]), min(old_bbox[1::2]),
                   max(old_bbox[0::2]), max(old_bbox[1::2])]
 
        best_match, best_score = None, 0
        for n_item in new_ocr:
            score = SequenceMatcher(None, old_text, n_item["text"]).ratio()
            if score > best_score:
                best_match, best_score = n_item, score
 
        if best_match and best_score >= 0.7:
            new_bbox = best_match["bbox"]
            new_box = [min(new_bbox[0::2]), min(new_bbox[1::2]),
                       max(new_bbox[0::2]), max(new_bbox[1::2])]
 
            ssim_score = compare_regions(old_img, new_img, old_box, new_box)
            if ssim_score < 0.75:
                pixel_differences.append({
                    "text": old_text,
                    "old_box": old_box,
                    "new_box": new_box,
                    "ssim_score": float(ssim_score),
                    "status": "VISUAL SHIFT / CHANGED"
                })
 
    return {
        "old_file": old_file.name,
        "new_file": new_file.name,
        "yolo_old": old_yolo,
        "yolo_new": new_yolo,
        "ocr_old": old_ocr,
        "ocr_new": new_ocr,
        "pixel_differences": pixel_differences  
    }
 
def get_latest_json() -> Path | None:
    json_files = sorted(glob.glob(str(UPLOAD_DIR / "*.json")), reverse=True)
    return Path(json_files[0]) if json_files else None
 
def save_results(results: list) -> Path:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = UPLOAD_DIR / f"combined_results_{timestamp}.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    return output_json
 
# ---------------------------
# Endpoints
# ---------------------------
# @app.post("/analyze")
# async def analyze_screenshots(old_files: List[UploadFile] = File(...), new_files: List[UploadFile] = File(...)):
#     """
#     Upload old & new screenshots, analyze via specialized agents.
#     """
#     #  Save uploaded files
#     saved_old = save_files(old_files, OLD_DIR)
#     saved_new = save_files(new_files, NEW_DIR)
 
#     #  Prepare agent input
#     agent_input = {
#         "old_files": [str(f) for f in saved_old],
#         "new_files": [str(f) for f in saved_new]
#     }
 
#     #  Invoke agent for comparison
#     try:
#         result = specialized_graph.invoke({"query": "compare_uploaded_files", "data": agent_input})
#     except Exception as e:
#         return {"error": str(e)}
 
#     #  Save results JSON
#     output_json = save_results(result.get("comparisons", []))
 
#     return {
#         "results": result.get("comparisons", []),
#         "saved_json": str(output_json),
#         "message": f"Analysis completed. {len(result.get('comparisons', []))} pairs processed."
#     }
#-------------------------------------------------------------------------------------------------
# @app.post("/analyze")
# async def analyze_screenshots(
#     old_files: List[UploadFile] = File(...),
#     new_files: List[UploadFile] = File(...),
# ):
#     from datetime import datetime
 
#     # 1. Save uploaded files
#     saved_old = await asyncio.to_thread(save_files, old_files, OLD_DIR)
#     saved_new = await asyncio.to_thread(save_files, new_files, NEW_DIR)
#     print(f"[INFO] Saved {len(saved_old)} old files and {len(saved_new)} new files.")
 
#     # 2. Pair files strictly by filename stem
#     pairs = pair_files(saved_old, saved_new)
#     print(f"[INFO] Found {len(pairs)} matching pairs.")
 
#     # 3. Analyze all pairs (YOLO, OCR, SSIM)
#     all_results = []
#     for old_file, new_file in pairs:
#         pair_result = await asyncio.to_thread(analyze_pair, old_file, new_file)
#         all_results.append(pair_result)
 
#     # 4. Save JSON
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_json = UPLOAD_DIR / f"combined_results_{timestamp}.json"
#     try:
#         with open(output_json, "w") as f:
#             json.dump(all_results, f, indent=2)
#         print(f"[INFO] JSON saved at {output_json}")
#     except Exception as e:
#         return {"error": "Failed to save JSON", "exception": str(e)}
 
#     return {
#         "results": all_results,
#         "saved_json": str(output_json),
#         "message": f"Analysis completed. {len(all_results)} pairs processed."
#     }
import os

@app.post("/analyze")
async def analyze_screenshots(
    old_files: List[UploadFile] = File(...),
    new_files: List[UploadFile] = File(...),
):
    from datetime import datetime

    # 1. Create a new folder for this upload session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = UPLOAD_DIR / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    old_session_dir = session_dir / "old"
    new_session_dir = session_dir / "new"
    old_session_dir.mkdir(exist_ok=True)
    new_session_dir.mkdir(exist_ok=True)

    # 2. Save uploaded files in session folders
    saved_old = await asyncio.to_thread(save_files, old_files, old_session_dir)
    saved_new = await asyncio.to_thread(save_files, new_files, new_session_dir)
    print(f"[INFO] Saved {len(saved_old)} old files and {len(saved_new)} new files in session {session_dir}")

    # 3. Pair files strictly by stem
    pairs = pair_files(saved_old, saved_new)
    print(f"[INFO] Found {len(pairs)} matching pairs.")

    # 4. Analyze each pair and save individual JSONs
    result_json_paths = []
    for old_file, new_file in pairs:
        pair_result = await asyncio.to_thread(analyze_pair, old_file, new_file)

        # Save JSON per pair
        pair_json_name = f"{old_file.stem}_vs_{new_file.stem}.json"
        pair_json_path = session_dir / pair_json_name
        with open(pair_json_path, "w") as f:
            json.dump(pair_result, f, indent=2)

        result_json_paths.append(str(pair_json_path))

    return {
        "message": f"Analysis completed. {len(pairs)} pairs processed.",
        "session_folder": str(session_dir),
        "json_files": result_json_paths
    }

# ---------------------------
from fastapi import FastAPI
import asyncio
from fastapi.responses import JSONResponse
from agent import extract_combined_discrepancies
@app.get("/generate_summary")
async def generate_summary():
    """
    Return all OCR, YOLO, and Pixel discrepancies from the latest session
    as a single flattened list, file-wise.
    """
    try:
        # Run the combined agent in a thread (avoid blocking)
        combined_results = await asyncio.to_thread(extract_combined_discrepancies)

        return JSONResponse(content={"combined_summary": combined_results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

 
# ---------------------------
 
@app.get("/download_report_pdf")
async def download_report_pdf():
    """
    Generate PDF report from latest JSON via agents.
    """
    latest_json = get_latest_json()
    if not latest_json:
        return PlainTextResponse("No JSON files found. Run /analyze first.", media_type="text/plain")
 
    # Ask agent to generate PDF
    try:
        result = specialized_graph.invoke({
            "query": "generate_pdf_report",
            "handle_parsing_errors": True,
            "data": str(latest_json)
        })
        pdf_path = Path(result.get("pdf_path", ""))
        if not pdf_path.exists():
            return PlainTextResponse("PDF generation failed.", media_type="text/plain")
    except Exception as e:
        return {"error": str(e)}
 
    return FileResponse(pdf_path, media_type="application/pdf", filename="discrepancy_report.pdf")
 
 