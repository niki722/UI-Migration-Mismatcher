import json
from pathlib import Path
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# ----------------------------------------------------
# Step 1: Load Combined JSON results
# ----------------------------------------------------
import json
from pathlib import Path

# ----------------------------------------------------
# Step 1: Load latest JSON results from uploads
# ----------------------------------------------------
uploads_dir = Path("uploads")
json_files = sorted(uploads_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

if not json_files:
    raise FileNotFoundError(" No JSON files found in uploads/ directory")

json_path = json_files[0]  # latest file
with open(json_path, "r", encoding="utf-8") as f:
    combined_results = json.load(f)

print(f" Loaded latest JSON: {json_path.name} with {len(combined_results)} entries")
print(f"First old file: {combined_results[0]['old_file']}")

# ----------------------------------------------------
# Step 2: Define tools
# ----------------------------------------------------
def get_discrepancy(location: str) -> str:
    return f"The discrepancy in {location} is significant."

def get_ocr_summary(index: int) -> str:
    """Fetch OCR comparison from combined_results JSON."""
    if index < 0 or index >= len(combined_results):
        return f"No OCR results for index {index}"
    
    result = combined_results[index]
    old_texts = [item["text"] for item in result["ocr_old"]]
    new_texts = [item["text"] for item in result["ocr_new"]]
    return f"OCR Old: {old_texts[:5]}...\nOCR New: {new_texts[:5]}..."

tools = [
    Tool(name="Discrepancy", func=get_discrepancy, description="Get discrepancy info"),
    Tool(name="OCRSummary", func=get_ocr_summary, description="Get OCR summary from JSON by index"),
]

# ----------------------------------------------------
# Step 3: Local LLM
# ----------------------------------------------------
llm = ChatOpenAI(
    base_url="http://172.52.50.82:3333/v1",  # your local LLM server
    api_key="not-needed",
    model="qwen2.5-vl-7b-instruct",
)

# ----------------------------------------------------
# Step 4: ReAct agent setup
# ----------------------------------------------------
agent_executor = create_react_agent(
    model=llm,
    tools=tools
)

# ----------------------------------------------------
# Step 5: LangGraph workflow
# ----------------------------------------------------
class AgentState(TypedDict):
    messages: list
    output: str

def agent_node(state: AgentState):
    if not state["messages"]:
        state["messages"] = []

    result = agent_executor.invoke({"messages": state["messages"]})
    state["messages"] = result["messages"]
    state["output"] = result["messages"][-1].content
    return state

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_edge("__start__", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# ----------------------------------------------------
# Step 6: Run test queries
# ----------------------------------------------------
queries = [
    "show ocr summary for index 0",
]

for q in queries:
    state = {"messages": [HumanMessage(content=q)], "output": ""}
    result = app.invoke(state)
    print(f"Query: {q}")
    print(f"Answer: {result['output']}\n{'-'*50}")  


# import json
# from pathlib import Path
# from typing import TypedDict
# from langchain_openai import ChatOpenAI
# from langchain.tools import Tool
# from langgraph.prebuilt import create_react_agent
# from langgraph.graph import StateGraph, END
# from langchain_core.messages import HumanMessage

# # ----------------------------------------------------
# # Step 1: Load Combined JSON results
# # ----------------------------------------------------
# import json
# from pathlib import Path

# # ----------------------------------------------------
# # Step 1: Load latest JSON results from uploads
# # ----------------------------------------------------
# uploads_dir = Path("uploads")
# json_files = sorted(uploads_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

# if not json_files:
#     raise FileNotFoundError(" No JSON files found in uploads/ directory")

# json_path = json_files[0]  # latest file
# with open(json_path, "r", encoding="utf-8") as f:
#     combined_results = json.load(f)

# print(f" Loaded latest JSON: {json_path.name} with {len(combined_results)} entries")
# print(f"First old file: {combined_results[0]['old_file']}")


# # ----------------------------------------------------
# # Step 2: Define UI Validation Tools
# # ----------------------------------------------------
# def check_color_contrast(index: int) -> str:
#     """Check color contrast issues for a given UI index."""
#     if index < 0 or index >= len(combined_results):
#         return f"No UI results for index {index}"

#     issues = combined_results[index].get("color_contrast", [])
#     return f"Color Contrast Issues: {issues if issues else 'None detected'}"

# def check_alignment(index: int) -> str:
#     """Check element alignment issues."""
#     if index < 0 or index >= len(combined_results):
#         return f"No UI results for index {index}"

#     issues = combined_results[index].get("alignment", [])
#     return f"Alignment Issues: {issues if issues else 'None detected'}"

# def check_responsiveness(index: int) -> str:
#     """Check responsiveness issues across devices."""
#     if index < 0 or index >= len(combined_results):
#         return f"No UI results for index {index}"

#     issues = combined_results[index].get("responsiveness", [])
#     return f"Responsiveness Issues: {issues if issues else 'None detected'}"

# def check_accessibility(index: int) -> str:
#     """Check accessibility issues like missing alt text or ARIA labels."""
#     if index < 0 or index >= len(combined_results):
#         return f"No UI results for index {index}"

#     issues = combined_results[index].get("accessibility", [])
#     return f"Accessibility Issues: {issues if issues else 'None detected'}"

# tools = [
#     Tool(name="ColorContrast", func=check_color_contrast, description="Check color contrast issues in UI"),
#     Tool(name="Alignment", func=check_alignment, description="Check element alignment issues in UI"),
#     Tool(name="Responsiveness", func=check_responsiveness, description="Check responsiveness issues across devices"),
#     Tool(name="Accessibility", func=check_accessibility, description="Check accessibility issues in UI"),
# ]

# # ----------------------------------------------------
# # Step 3: Local LLM
# # ----------------------------------------------------
# llm = ChatOpenAI(
#     base_url="http://172.52.50.82:3333/v1",  # your local LLM server
#     api_key="not-needed",
#     model="qwen2.5-vl-7b-instruct",
# )

# # ----------------------------------------------------
# # Step 4: ReAct Agent
# # ----------------------------------------------------
# agent_executor = create_react_agent(
#     model=llm,
#     tools=tools
# )

# # ----------------------------------------------------
# # Step 5: LangGraph workflow
# # ----------------------------------------------------
# class AgentState(TypedDict):
#     messages: list
#     output: str

# def agent_node(state: AgentState):
#     if not state["messages"]:
#         state["messages"] = []

#     result = agent_executor.invoke({"messages": state["messages"]})
#     state["messages"] = result["messages"]
#     state["output"] = result["messages"][-1].content
#     return state

# workflow = StateGraph(AgentState)
# workflow.add_node("agent", agent_node)
# workflow.add_edge("__start__", "agent")
# workflow.add_edge("agent", END)

# app = workflow.compile()

# # ----------------------------------------------------
# # Step 6: Run test queries
# # ----------------------------------------------------
# queries = [
#     "Check color contrast issues for screen index 0",
#     "Are there any alignment issues in screen 2?",
#     "Validate responsiveness in index 1",
#     "What accessibility problems exist in index 3?"
# ]

# for q in queries:
#     state = {"messages": [HumanMessage(content=q)], "output": ""}
#     result = app.invoke(state)
#     print(f"Query: {q}")
#     print(f"Answer: {result['output']}\n{'-'*50}")