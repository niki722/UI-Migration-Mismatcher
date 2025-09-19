# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import shutil
import json
from difflib import SequenceMatcher
import cv2
import easyocr
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import glob
from difflib import SequenceMatcher
from fastapi import FastAPI
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from fastapi.responses import FileResponse, PlainTextResponse
import asyncio
# Suppress CancelledError in asyncio event loop
asyncio.get_event_loop().set_exception_handler(
    lambda loop, context: None if isinstance(context.get("exception"), asyncio.CancelledError) else loop.default_exception_handler(context)
)

# ---------------------------
# Setup
# ---------------------------
app = FastAPI(title="FeatureMirror - Enhanced OCR + YOLO Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
from skimage.metrics import structural_similarity as ssim
import numpy as np
def format_displacement(diff, pixel_diff=None):
    """
    diff: dict from position change (avg delta)
    pixel_diff: dict from pixel/SSIM change
    """
    label = diff.get("text") or diff.get("element") or "Unknown"

    # Position change info
    pos_delta = diff.get("position_delta") or diff.get("avg_delta")
    pos_str = f"Position changed: avg delta {pos_delta:.1f}px" if pos_delta else ""

    # Pixel similarity info
    ssim_score = pixel_diff.get("ssim_score") if pixel_diff else None
    status = pixel_diff.get("status") if pixel_diff else ""
    if ssim_score is not None:
        if ssim_score < 0.3:
            severity = "Major change"
        elif ssim_score < 0.6:
            severity = "Moderate change"
        else:
            severity = "Minor shift"
        pixel_str = f"{status.replace('/', ' ').title()} ({severity}, similarity: {ssim_score:.2f})"
    else:
        pixel_str = ""

    # Combine
    combined = label
    if pos_str and pixel_str:
        combined += f": {pos_str}, {pixel_str}"
    elif pos_str:
        combined += f": {pos_str}"
    elif pixel_str:
        combined += f": {pixel_str}"

    return combined

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

# Temporary upload folders
UPLOAD_DIR = Path("uploads")
OLD_DIR = UPLOAD_DIR / "old"
NEW_DIR = UPLOAD_DIR / "new"
OLD_DIR.mkdir(parents=True, exist_ok=True)
NEW_DIR.mkdir(parents=True, exist_ok=True)

# Output JSON path
OUTPUT_JSON = UPLOAD_DIR / "combined_results.json"

# ---------------------------
# Initialize OCR, YOLO
# ---------------------------
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
        "pixel_differences": pixel_differences  # now will be populated
    }


# ---------------------------
# Endpoint
# ---------------------------
@app.post("/analyze")
async def analyze_screenshots(
    old_files: List[UploadFile] = File(...),
    new_files: List[UploadFile] = File(...),
):
    from datetime import datetime

    # 1️⃣ Save uploaded files
    saved_old = save_files(old_files, OLD_DIR)
    saved_new = save_files(new_files, NEW_DIR)

    print(f"[INFO] Saved {len(saved_old)} old files and {len(saved_new)} new files.")

    # 2️⃣ Pair files strictly by filename stem
    pairs = pair_files(saved_old, saved_new)
    print(f"[INFO] Found {len(pairs)} matching pairs.")

    all_results = []

    # 3️⃣ Analyze only paired files
    if pairs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(analyze_pair, old, new): (old, new) for old, new in pairs}
            for future in as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    old_file, new_file = futures[future]
                    print(f"[ERROR] Failed analyzing {old_file.name} → {new_file.name}: {e}")
    else:
        print("[WARNING] No matching pairs found. JSON will still be created with empty results.")

    # 4️⃣ Save JSON with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_name = f"combined_results_{timestamp}.json"
    OUTPUT_JSON = UPLOAD_DIR / output_json_name

    try:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[INFO] JSON saved at {OUTPUT_JSON}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")
        return {"error": "Failed to save JSON", "exception": str(e)}

    # 5️⃣ Return results
    return {
        "results": all_results,
        "saved_json": str(OUTPUT_JSON),
        "message": f"Analysis completed. {len(all_results)} pairs processed."
    }


# ---------------------------
# Initialize LLM
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

UPLOAD_DIR = Path("uploads")

# ---------------------------
# New route for generating discrepancy summary from latest JSON
# ---------------------------
@app.get("/generate_summary")
async def generate_summary():
    import glob

    # Find latest JSON
    json_files = sorted(glob.glob(str(UPLOAD_DIR / "*.json")), reverse=True)
    if not json_files:
        return {"error": "No JSON files found. Run /analyze first."}

    latest_json_path = Path(json_files[0])
    with open(latest_json_path, "r") as f:
        all_results = json.load(f)

    summary_lines = []

    def bbox_distance(b1, b2):
        # b1 and b2 are lists of 8 floats (4 points x,y)
        if not b1 or not b2 or len(b1) != 8 or len(b2) != 8:
            return None
        # Calculate average distance between corresponding points
        return sum(abs(b1[i] - b2[i]) for i in range(8)) / 8

    for pair in all_results:
        if isinstance(pair, str):  # handle double-encoded JSON just in case
            try:
                pair = json.loads(pair)
            except Exception:
                continue  

        if not isinstance(pair, dict):
            continue

        old_file = pair.get("old_file", "old_file")
        new_file = pair.get("new_file", "new_file")
        summary_lines.append(f"{old_file} → {new_file}")

        ocr_old = pair.get("ocr_old", [])
        ocr_new = pair.get("ocr_new", [])

        for o_item in ocr_old:
            best_match, best_score, best_bbox = None, 0, None
            for n_item in ocr_new:
                score = SequenceMatcher(None, o_item["text"], n_item["text"]).ratio()
                if score > best_score:
                    best_score, best_match, best_bbox = score, n_item, n_item.get("bbox")

            dist = 0
            if best_score >= 0.95 and best_bbox:
                # small position change is still pass
                if bbox_distance(o_item.get("bbox"), best_bbox) > 20:
                    dist = bbox_distance(o_item.get("bbox"), best_bbox)
                line = format_displacement(
                    {"text": o_item["text"], "avg_delta": dist if dist else 0},
                    next((d for d in pair.get("pixel_differences", []) if d.get("text") == o_item["text"]), None)
                )
                summary_lines.append(f"PASSED → {line}")
            else:
                found_text = best_match["text"] if best_match else "Not Found"
                similarity = round(best_score * 100, 2)
                summary_lines.append(f"FAILED → {o_item['text']}: Text changed to '{found_text}' (similarity: {similarity}%)")

        # YOLO elements
        yolo_old = pair.get("yolo_old", [])
        yolo_new = pair.get("yolo_new", [])
        for e_old in yolo_old:
            elem = e_old.get("class_name", str(e_old.get("class_id")))
            matches = [e for e in yolo_new if e.get("class_name", str(e.get("class_id"))) == elem]
            if matches:
                old_box = e_old.get("xyxy")
                new_box = matches[0].get("xyxy")
                pos_delta = sum(abs(old_box[i] - new_box[i]) for i in range(4)) / 4 if old_box and new_box else 0
                summary_lines.append(f"PASSED → {elem}" if pos_delta <= 20 else f"FAILED → {elem}: Position changed (avg delta {pos_delta:.1f}px)")
            else:
                summary_lines.append(f"FAILED → {elem}: Element missing in new UI")



        summary_lines.append("")  # Empty line between pairs

    # Use LLM to summarize the results
    llm_prompt = "Summarize the following UI migration results as bullet points, highlighting all failures and warnings, and giving a brief overall assessment at the end.\n" + "\n".join(summary_lines)
    llm_summary = llm_pipe(llm_prompt, max_length=512)[0]['generated_text']

    return {
        "latest_json": str(latest_json_path),
        "summary": "\n".join(summary_lines),
        "llm_summary": llm_summary
    }


@app.get("/download_report_pdf")
async def download_report_pdf():
    # Find latest JSON
    json_files = sorted(glob.glob(str(UPLOAD_DIR / "*.json")), reverse=True)
    if not json_files:
        return PlainTextResponse("No JSON files found. Run /analyze first.", media_type="text/plain")

    latest_json_path = Path(json_files[0])
    with open(latest_json_path, "r") as f:
        all_results = json.load(f)

    # PDF path
    pdf_path = UPLOAD_DIR / "discrepancy_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = 1  # center
    elements.append(Paragraph("UI Migration Discrepancy Report", title_style))
    elements.append(Spacer(1, 20))

    # Global counters
    global_pass, global_warn, global_fail, global_total = 0, 0, 0, 0

    for pair in all_results:
        if not pair.get("old_file") or not pair.get("new_file"):
            continue

        old_file = pair.get("old_file", "old_file")
        new_file = pair.get("new_file", "new_file")

        # File Header
        file_header_style = styles["Heading2"]
        elements.append(Paragraph(f"{old_file} → {new_file}", file_header_style))
        elements.append(Spacer(1, 10))

        # Table setup
        table_data = [["Status", "Old Text/Element", "New Text/Element", "Similarity"]]
        row_styles = []

        # File-level counters
        pass_count, warn_count, fail_count, total_count = 0, 0, 0, 0

        # OCR comparisons
        ocr_old = pair.get("ocr_old", [])
        ocr_new = pair.get("ocr_new", [])

        for o_item in ocr_old:
            best_match, best_score = None, 0
            for n_item in ocr_new:
                score = SequenceMatcher(None, o_item["text"], n_item["text"]).ratio()
                if score > best_score:
                    best_score, best_match = score, n_item

            text_old = o_item["text"]
            text_new = best_match["text"] if best_match else "Not Found"
            similarity = f"{round(best_score*100, 2)}%"

            if best_score >= 0.95:
                status = "PASS"
                row_color = colors.lightgreen
                pass_count += 1
            elif best_score >= 0.7:
                status = "WARNING"
                row_color = colors.lightyellow
                warn_count += 1
            else:
                status = "FAIL"
                row_color = colors.salmon
                fail_count += 1

            total_count += 1
            table_data.append([status, text_old, text_new, similarity])
            row_styles.append(("BACKGROUND", (0, len(table_data)-1), (-1, len(table_data)-1), row_color))

        # YOLO comparisons
        yolo_old = [e.get("class_name", str(e.get("class_id"))) for e in pair.get("yolo_old", [])]
        yolo_new = [e.get("class_name", str(e.get("class_id"))) for e in pair.get("yolo_new", [])]

        for elem in yolo_old:
            if elem in yolo_new:
                status, text_new, similarity, row_color = "PASS", elem, "100%", colors.lightgreen
                pass_count += 1
            else:
                status, text_new, similarity, row_color = "FAIL", "Missing", "0%", colors.salmon
                fail_count += 1

            total_count += 1
            table_data.append([status, elem, text_new, similarity])
            row_styles.append(("BACKGROUND", (0, len(table_data)-1), (-1, len(table_data)-1), row_color))

        # Update global counters
        global_pass += pass_count
        global_warn += warn_count
        global_fail += fail_count
        global_total += total_count

        # Create table
        table = Table(table_data, colWidths=[80, 180, 180, 80])
        table_style = TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ] + row_styles)

        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 12))

        # File-level summary
        elements.append(Paragraph(
            f"Summary: Total = {total_count}, PASS = {pass_count}, WARNING = {warn_count}, FAIL = {fail_count}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 20))

    # Overall summary
    elements.append(Paragraph("Overall Project Summary", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Files Analyzed = {len(all_results)}<br/>"
        f"Total Checks = {global_total}<br/>"
        f"PASS = {global_pass}, WARNING = {global_warn}, FAIL = {global_fail}",
        styles["Normal"]
    ))

    # Build PDF
    doc.build(elements)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="discrepancy_report.pdf"
    )
