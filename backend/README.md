FeatureMirror Backend

🔍 FeatureMirror is a backend service for automated UI migration validation.
It analyzes old vs. new screenshots using YOLOv8 (UI element detection), EasyOCR (text recognition), and Flan-T5-small (summarization), then produces structured JSON reports and human-readable summaries.

🚀 Quick Start
1. Clone & Install
git clone <your-repo-url>
cd featuremirror-backend
pip install -r requirements.txt

2. Run the API Server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 5000


API base URL: http://127.0.0.1:5000

Interactive docs: http://127.0.0.1:5000/docs

📡 API Endpoints
🔹 1. Analyze Screenshots

POST /analyze
Upload old & new screenshots for automated comparison.

curl -X POST "http://127.0.0.1:5000/analyze" \
  -F "old_files=@old_ui1.png" \
  -F "old_files=@old_ui2.png" \
  -F "new_files=@new_ui1.png" \
  -F "new_files=@new_ui2.png"


📌 What happens:

Screenshots saved under uploads/old/ and uploads/new/

Old ↔ New files auto-paired by filename similarity

YOLOv8 detects UI components

EasyOCR extracts text with confidence filtering

JSON results written to uploads/combined_results_<timestamp>.json

🔹 2. Generate Summary

GET /generate_summary
Generate a human-readable summary from the latest analysis.

curl -X GET "http://127.0.0.1:5000/generate_summary"


📌 Response Example:

{
  "latest_json": "uploads/combined_results_20250917_153045.json",
  "summary": "3.webp → 3.webp\nPASSED → TheCubeFactory\nPASSED → Welcome back\nFAILED → Sign in: Text changed to 'Login' (similarity: 68%)\nFAILED → button: Element missing in new UI\n"
}

⚙️ How It Works

Feature Detection

YOLOv8 detects buttons, inputs, and UI regions

EasyOCR extracts text with bounding boxes

Auto-Pairing

Old and new screenshots matched by filename similarity

Analysis

Text similarity measured with difflib.SequenceMatcher

Missing/mismatched UI components flagged

Summarization

CPU-friendly flan-t5-small generates reports

🛠 Tech Stack

FastAPI → API framework

YOLOv8 → UI element detection

EasyOCR → OCR text extraction

OpenCV → Preprocessing

Flan-T5-small (Transformers) → Natural language summaries

difflib → Text similarity

📊 Performance

CPU-only (no GPU required)

~2–4 cores used during analysis

~1–2 GB RAM per batch

5–10 screenshots processed per minute

🔧 Configuration

Optional environment variables:

# Model settings
LLM_MODEL_NAME=google/flan-t5-small

# Thresholds
YOLO_CONF_THRESH=0.3
OCR_CONF_THRESH=0.5
SIMILARITY_THRESHOLD=0.7

# Storage paths
UPLOAD_DIR=uploads

🐛 Troubleshooting

❌ 404 Not Found
Check that you’re posting to /analyze (not /upload).

❌ OCR results inaccurate

Use high-resolution screenshots

Increase contrast or scale images

❌ YOLO missing elements

Adjust YOLO_CONF_THRESH

Ensure training covers your UI dataset

🧪 Development
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black isort

# Run tests
pytest

# Format code
black .
isort .

📜 License

MIT License – See LICENSE file for details.

📬 Support

📚 API Docs: http://127.0.0.1:5000/docs

🐛 Report issues via GitHub

📧 Contact: team@featuremirror.dev