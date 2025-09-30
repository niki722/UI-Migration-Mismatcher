from difflib import SequenceMatcher
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from utils import load_latest_results
 
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
# ---------------------------
# LLM Initialization
# ---------------------------
llm = ChatOpenAI(
    base_url="http://172.52.50.82:3333/v1",  
    api_key="not-needed",
    model="qwen2.5-vl-7b-instruct",
)
 
# ---------------------------
def fallback_agent(query: str):
    return f"Sorry, I can’t classify query: {query}"
 
import asyncio
 
async def consume_queue(queue: asyncio.Queue):
    try:
        while True:
            item = await queue.get()
            # Process item
            print(item)
            queue.task_done()
    except asyncio.CancelledError:
        print("Queue consumer task was cancelled, exiting gracefully.")
       
from pathlib import Path
import json
import glob

UPLOAD_DIR = Path("uploads")

def load_latest_results():
    # Find all session folders
    session_folders = sorted([f for f in UPLOAD_DIR.iterdir() if f.is_dir() and f.name.startswith("session_")], reverse=True)
    if not session_folders:
        return "No session folders found."

    latest_session = session_folders[0]
    json_files = sorted(latest_session.glob("*.json"))
    if not json_files:
        return "No JSON files found in latest session."

    all_results = []
    for jf in json_files:
        with open(jf, "r") as f:
            try:
                data = json.load(f)
                all_results.append(data)
            except Exception as e:
                print(f"Failed to load {jf}: {e}")

    # Flatten the list of JSONs (each JSON contains one pair)
    flattened_results = [item for sublist in all_results for item in (sublist if isinstance(sublist, list) else [sublist])]
    return flattened_results
 
# ---------------------------
# Extract Functions
# ---------------------------
 
def extract_ocr_discrepancies():
    results = load_latest_results()
    if isinstance(results, str):
        return results
 
    output = []
    for pair in results:
        old_file = pair.get("old_file", "")
        new_file = pair.get("new_file", "")
        diffs = []
 
        for o_item in pair.get("ocr_old", []):
            match = next(
                (n for n in pair.get("ocr_new", [])
                 if SequenceMatcher(None, o_item["text"], n["text"]).ratio() > 0.7),
                None,
            )
            if not match:
                diffs.append(f"{o_item['text']} missing in {new_file}")
 
        if diffs:
            output.append({
                "old_file": old_file,
                "new_file": new_file,
                "type": "OCR",
                "messages": diffs
            })
 
    return output if output else [{"message": "No OCR discrepancies."}]
 
def extract_yolo_discrepancies():
    results = load_latest_results()
    if isinstance(results, str):
        return results
 
    output = []
    for pair in results:
        old_file = pair.get("old_file", "")
        new_file = pair.get("new_file", "")
        yolo_old = [e.get("class_name") for e in pair.get("yolo_old", [])]
        yolo_new = [e.get("class_name") for e in pair.get("yolo_new", [])]
        missing = [e for e in yolo_old if e not in yolo_new]
 
        if missing:
            output.append({
                "old_file": old_file,
                "new_file": new_file,
                "type": "YOLO",
                "messages": [f"Missing: {', '.join(missing)}"]
            })
 
    return output if output else [{"message": "No YOLO discrepancies."}]
 
 
 
def extract_statistics():
    results = load_latest_results()
    if isinstance(results, str):
        return results
 
    total, fail, warn, passed = 0, 0, 0, 0
    for pair in results:
        total += len(pair.get("ocr_old", [])) + len(pair.get("yolo_old", []))
        for diff in pair.get("pixel_differences", []):
            ssim_score = diff.get("ssim_score", 1)
            if ssim_score < 0.3:
                fail += 1
            elif ssim_score < 0.6:
                warn += 1
            else:
                passed += 1
 
    return {
        "total": total,
        "pass": passed,
        "warn": warn,
        "fail": fail
    }
 
 
def extract_pixel_discrepancies():
    results = load_latest_results()
    if isinstance(results, str):
        return results
 
    output = []
    for pair in results:
        old_file = pair.get("old_file", "")
        new_file = pair.get("new_file", "")
        diffs = []
 
        for diff in pair.get("pixel_differences", []):
            label = diff.get("text") or diff.get("element") or "Unknown"
            ssim_score = diff.get("ssim_score", 0)
            status = diff.get("status", "UNKNOWN")
            diffs.append(f"{label}: {status} (similarity: {ssim_score:.2f})")
 
        if diffs:
            output.append({
                "old_file": old_file,
                "new_file": new_file,
                "type": "PIXEL",
                "messages": diffs
            })
 
    return output if output else [{"message": "No pixel differences."}]
 
 
def extract_combined_discrepancies():
    """
    Returns all discrepancies from latest session JSONs as a single flattened list.
    Each entry includes file pair, type, and message.
    """
    results = load_latest_results()
    if isinstance(results, str):
        return [{"message": results}]

    combined = []

    for pair in results:
        old_file = pair.get("old_file", "")
        new_file = pair.get("new_file", "")

        # --- OCR discrepancies ---
        for o_item in pair.get("ocr_old", []):
            match = next(
                (n for n in pair.get("ocr_new", [])
                 if SequenceMatcher(None, o_item["text"], n["text"]).ratio() > 0.7),
                None
            )
            if not match:
                combined.append({
                    "file_pair": f"{old_file} vs {new_file}",
                    "type": "OCR",
                    "message": f"'{o_item['text']}' missing in new file"
                })

        # --- YOLO discrepancies ---
        yolo_old = [e.get("class_name") for e in pair.get("yolo_old", [])]
        yolo_new = [e.get("class_name") for e in pair.get("yolo_new", [])]
        for e in yolo_old:
            if e not in yolo_new:
                combined.append({
                    "file_pair": f"{old_file} vs {new_file}",
                    "type": "YOLO",
                    "message": f"Element '{e}' missing in new file"
                })

        # --- Pixel / SSIM differences ---
        for diff in pair.get("pixel_differences", []):
            label = diff.get("text") or diff.get("element") or "Unknown"
            ssim_score = diff.get("ssim_score", 0)
            status = diff.get("status", "UNKNOWN")
            combined.append({
                "file_pair": f"{old_file} vs {new_file}",
                "type": "PIXEL",
                "message": f"{label}: {status} (similarity: {ssim_score:.2f})"
            })

    if not combined:
        combined.append({"message": "No discrepancies found in latest session."})

    return combined
 
def route_query(state: Dict[str, Any]):
    q = state["query"].lower()
    if "ocr" in q or "text" in q:
        return {"answer": ocr_agent.run("Extract OCR discrepancies")}
    elif "yolo" in q or "element" in q or "ui" in q:
        return {"answer": yolo_agent.run("Extract YOLO discrepancies")}
    elif "pixel" in q or "ssim" in q:
        return {"answer": pixel_agent.run("Extract pixel discrepancies")}
    elif "stat" in q or "summary" in q or "overall" in q:
        return {"answer": stats_agent.run("Summarize overall statistics")}
    else:
        return {"answer": fallback_agent(q)}
 
# ---------------------------
# Tools + Specialized Agents
# ---------------------------
 
ocr_tool = Tool(
    name="OCRDiscrepancies",
    func=lambda _: extract_ocr_discrepancies(),
    description="Get only OCR-related discrepancies."
)
yolo_tool = Tool(
    name="YOLODiscrepancies",
    func=lambda _: extract_yolo_discrepancies(),
    description="Get only YOLO-related discrepancies."
)
stats_tool = Tool(
    name="StatsSummary",
    func=lambda _: extract_statistics(),
    description="Get global summary stats of PASS/FAIL/WARN."
)
pixel_tool = Tool(
    name="PixelDiscrepancies",
    func=lambda _: extract_pixel_discrepancies(),
    description="Get pixel/SSIM differences for UI elements and text."
)
combined_tool = Tool(
    name="CombinedDiscrepancies",
    func=lambda _: extract_combined_discrepancies(),
    description="Return all OCR, YOLO, and Pixel discrepancies from latest session as a flattened list."
)

combined_agent = initialize_agent(
    [combined_tool],
    llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

ocr_agent = initialize_agent([ocr_tool], llm, agent="structured-chat-zero-shot-react-description", verbose=True,handle_parsing_errors=True)
yolo_agent = initialize_agent([yolo_tool], llm, agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)
stats_agent = initialize_agent([stats_tool], llm, agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)
pixel_agent = initialize_agent([pixel_tool], llm, agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)
 
 
# ---------------------------
# Coordinator Graph
# ---------------------------
 
def build_specialized_graph():
    graph = StateGraph(dict)
 
    def route_query(state: Dict[str, Any]):
        q = state["query"].lower()
        if "ocr" in q or "text" in q:
            return {"answer": ocr_agent.run(q)}
        elif "yolo" in q or "element" in q or "ui" in q:
            return {"answer": yolo_agent.run(q)}
        elif "pixel" in q or "ssim" in q:
            return {"answer": pixel_agent.run(q)}
        elif "stat" in q or "summary" in q or "overall" in q:
            return {"answer": stats_agent.run(q)}
        else:
            return {"answer": fallback_agent(q)}
 
    graph.add_node("router", route_query)
    graph.set_entry_point("router")
    graph.add_edge("router", END)
    return graph.compile()
 
specialized_graph = build_specialized_graph()
 
 