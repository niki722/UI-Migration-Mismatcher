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
    """Pair files by filename similarity."""
    pairs = []
    for old in old_files:
        best_match = None
        best_ratio = 0
        for new in new_files:
            ratio = SequenceMatcher(None, old.stem, new.stem).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = new
        if best_match:
            pairs.append((old, best_match))
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

def analyze_pair(old_file: Path, new_file: Path):
    """Analyze a single pair of screenshots thoroughly."""
    old_yolo = run_yolo(old_file)
    new_yolo = run_yolo(new_file)
    old_ocr = run_ocr(old_file)
    new_ocr = run_ocr(new_file)

    return {
        "old_file": old_file.name,
        "new_file": new_file.name,
        "yolo_old": old_yolo,
        "yolo_new": new_yolo,
        "ocr_old": old_ocr,
        "ocr_new": new_ocr
    }

# ---------------------------
# Endpoint
# ---------------------------
@app.post("/analyze")
async def analyze_screenshots(
    old_files: List[UploadFile] = File(...),
    new_files: List[UploadFile] = File(...)
):
    # 1️ Save uploaded files locally
    saved_old = save_files(old_files, OLD_DIR)
    saved_new = save_files(new_files, NEW_DIR)

    # 2️ Auto pair old & new
    pairs = pair_files(saved_old, saved_new)
    all_results = []

    if not pairs:
        return {"error": "No matching screenshot pairs found. Check filenames."}

    # 3️ Analyze each pair in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_pair, old, new): (old, new) for old, new in pairs}
        for future in as_completed(futures):
            all_results.append(future.result())

    # 4️ Save combined JSON per pair
        from datetime import datetime

# Inside the endpoint after analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_name = f"combined_results_{timestamp}.json"
        OUTPUT_JSON = UPLOAD_DIR / output_json_name

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    return {"results": all_results, "saved_json": str(OUTPUT_JSON)}


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
            best_match = None
            best_score = 0
            best_bbox = None
            for n_item in ocr_new:
                score = SequenceMatcher(None, o_item["text"], n_item["text"]).ratio()
                if score > best_score:
                    best_score = score
                    best_match = n_item
                    best_bbox = n_item.get("bbox")

            position_changed = False
            pos_change_str = ""
            if best_score >= 0.7 and best_bbox:
                dist = bbox_distance(o_item.get("bbox"), best_bbox)
                if dist is not None and dist > 20:  # threshold for position change
                    position_changed = True
                    pos_change_str = f" [Position changed: avg delta {dist:.1f}px]"

            if best_score >= 0.7:
                status = "PASSED" if not position_changed else "WARNING"
                summary_lines.append(f"{status} → {o_item['text']}{pos_change_str}")
            else:
                found_text = best_match["text"] if best_match else "Not Found"
                similarity = round(best_score * 100, 2)
                summary_lines.append(f"FAILED → {o_item['text']}: Text changed to '{found_text}' (similarity: {similarity}%)")

        # YOLO element checks
        yolo_old = pair.get("yolo_old", [])
        yolo_new = pair.get("yolo_new", [])
        yolo_new_names = [e.get("class_name", str(e.get("class_id"))) for e in yolo_new]

        for e_old in yolo_old:
            elem = e_old.get("class_name", str(e_old.get("class_id")))
            # Find matching element in new by class name
            matches = [e for e in yolo_new if e.get("class_name", str(e.get("class_id"))) == elem]
            if matches:
                # Compare box positions
                old_box = e_old.get("xyxy")
                new_box = matches[0].get("xyxy")
                pos_delta = None
                if old_box and new_box and len(old_box) == 4 and len(new_box) == 4:
                    pos_delta = sum(abs(old_box[i] - new_box[i]) for i in range(4)) / 4
                if pos_delta and pos_delta > 20:
                    summary_lines.append(f"WARNING → {elem}: Element position changed (avg delta {pos_delta:.1f}px)")
                else:
                    summary_lines.append(f"PASSED → {elem}: Element preserved")
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

    # PDF file path
    pdf_path = UPLOAD_DIR / "discrepancy_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("UI Migration Discrepancy Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Global counters
    global_pass, global_warn, global_fail, global_total = 0, 0, 0, 0

    for pair in all_results:
        old_file = pair.get("old_file", "old_file")
        new_file = pair.get("new_file", "new_file")

        elements.append(Paragraph(f"<b>{old_file} → {new_file}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Build table rows
        table_data = [["Status", "Old Text/Element", "New Text/Element", "Similarity"]]
        row_styles = []  # dynamic row coloring

        # File-level counters
        pass_count, warn_count, fail_count, total_count = 0, 0, 0, 0

        # OCR check
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
                status = "✅ PASS"
                row_color = colors.lightgreen
                pass_count += 1
            elif best_score >= 0.7:
                status = " WARNING"
                row_color = colors.lightyellow
                warn_count += 1
            else:
                status = " FAIL"
                row_color = colors.salmon
                fail_count += 1

            total_count += 1
            table_data.append([status, text_old, text_new, similarity])
            row_styles.append(("BACKGROUND", (0, len(table_data)-1), (-1, len(table_data)-1), row_color))

        # YOLO check
        yolo_old = [e.get("class_name", str(e.get("class_id"))) for e in pair.get("yolo_old", [])]
        yolo_new = [e.get("class_name", str(e.get("class_id"))) for e in pair.get("yolo_new", [])]

        for elem in yolo_old:
            if elem in yolo_new:
                status, text_new, similarity, row_color = " PASS", elem, "100%", colors.lightgreen
                pass_count += 1
            else:
                status, text_new, similarity, row_color = " FAIL", "Missing", "0%", colors.salmon
                fail_count += 1

            total_count += 1
            table_data.append([status, elem, text_new, similarity])
            row_styles.append(("BACKGROUND", (0, len(table_data)-1), (-1, len(table_data)-1), row_color))

        # Update global counts
        global_pass += pass_count
        global_warn += warn_count
        global_fail += fail_count
        global_total += total_count

        # Create table
        table = Table(table_data, colWidths=[80, 150, 150, 80])
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

        # File summary
        elements.append(Paragraph(
            f"<b>Summary:</b> Total = {total_count},  PASS = {pass_count},  WARNING = {warn_count}, ❌ FAIL = {fail_count}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 20))

    # Global summary at end
    elements.append(Paragraph("<b>Overall Project Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Files Analyzed = {len(all_results)}<br/>"
        f"Total Checks = {global_total}<br/>"
        f" PASS = {global_pass},  WARNING = {global_warn}, ❌ FAIL = {global_fail}",
        styles["Normal"]
    ))

    doc.build(elements)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="discrepancy_report.pdf"
    )


