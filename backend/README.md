# FeatureMirror - UI Migration Analyzer

FeatureMirror is a FastAPI-based tool to automatically compare screenshots of web or mobile UIs. It leverages **OCR**, **YOLO object detection**, and **SSIM-based visual comparison** to detect text changes, layout shifts, and visual differences between old and new versions of the UI.

---

## Features

- **OCR Text Analysis**: Detects text changes between old and new screenshots.
- **YOLO Object Detection**: Detects UI components and identifies missing or shifted elements.
- **Pixel-level Comparison**: Uses SSIM to detect visual shifts or changes.
- **Automated JSON Output**: Stores detailed analysis results for further processing.
- **PDF Report Generation**: Creates a professional PDF report with file-level and overall summaries.
- **LLM Summary Generation**: Uses a T5-based model to summarize analysis results in human-readable bullet points.
- **FastAPI Endpoints**:
  - `/analyze` – Upload old and new screenshots for analysis.
  - `/generate_summary` – Generate a textual summary of differences.
  - `/download_report_pdf` – Download a detailed PDF discrepancy report.

---

## Installation

1. Clone this repository:

  ```bash
  git clone <repo-url>
  cd <repo-folder>

2. Install dependencies:

  pip install -r requirements.txt

3. Usage

  --> Run the FastAPI server:

  --> python -m uvicorn main:app --reload --host 127.0.0.1 --port 5000


4. Open the API docs in your browser:

http://127.0.0.1:8000/docs


5. Use the endpoints:

  /analyze: Upload old and new UI screenshots (paired by filename stem).

  /generate_summary: Get a textual summary of detected changes.

  /download_report_pdf: Download the PDF report.

6. Example Workflow
    1) Upload old and new screenshots of your UI.

    2) Analyze differences via /analyze.

    3) View summarized textual report via /generate_summary.

    4)Download detailed PDF report via /download_report_pdf.

5. Notes

~ Make sure yolov8n.pt or your trained YOLO model is in the working directory.

~ OCR may fail for very small or low-resolution text.

~ PDF report includes colored rows to indicate pass (green), warning (yellow), and fail (red) results.