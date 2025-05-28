"""
API module for the Research Assistant Network.
"""
from typing import Dict, List, Optional
import os
import uuid
import asyncio
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
import threading

from langchain_openai import ChatOpenAI
from langchain_community.tools import BaseTool
from langchain.tools.python.tool import PythonREPLTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_openai import OpenAIEmbeddings

from src.config.config import API_HOST, API_PORT, API_DEBUG, OPENAI_API_KEY, STORAGE_DIR
from src.orchestration import run_research_pipeline, AgentState


# Initialize Flask app
app = Flask(__name__)

# Dictionary to store task status and results
tasks = {}

# Configure OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize language model
llm = ChatOpenAI(temperature=0.2, model="gpt-4")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize tools
def get_tools():
    """Get tools for each agent."""
    # Common tools
    python_tool = PythonREPLTool()
    
    # Info gatherer tools
    try:
        tavily_search = TavilySearchResults()
    except:
        tavily_search = None
    
    try:
        arxiv_tool = ArxivQueryRun()
    except:
        arxiv_tool = None
    
    try:
        semantic_scholar_tool = SemanticScholarQueryRun()
    except:
        semantic_scholar_tool = None
    
    # Organize tools by agent
    tools = {
        "info_gatherer": [t for t in [tavily_search, arxiv_tool, semantic_scholar_tool] if t is not None],
        "insight_generator": [],
        "data_analyst": [python_tool],
        "hypothesis_tester": [python_tool],
        "report_compiler": [],
    }
    
    return tools


# Progress callback
def update_task_status(task_id, event, state):
    """Update task status based on graph events."""
    current_node = event.get("node") if isinstance(event, dict) else str(event)
    
    status_mapping = {
        "info_gatherer": "Gathering information",
        "insight_generator": "Generating insights",
        "data_analyst": "Analyzing data",
        "hypothesis_tester": "Testing hypotheses",
        "report_compiler": "Compiling report",
        "END": "Completed",
    }
    
    status = status_mapping.get(current_node, "Processing")
    
    tasks[task_id]["status"] = status
    tasks[task_id]["last_updated"] = datetime.now().isoformat()
    
    if current_node == "END":
        tasks[task_id]["completed"] = True
        tasks[task_id]["results"] = {
            "final_report": state.get("final_report", {}),
            "research_query": state.get("research_query", ""),
        }


# API routes
@app.route("/")
def index():
    """Render the web interface."""
    return render_template("index.html")


@app.route("/api/research", methods=["POST"])
def start_research():
    """Start a new research task."""
    data = request.json
    research_query = data.get("query")
    
    if not research_query:
        return jsonify({"error": "Research query is required"}), 400
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    tasks[task_id] = {
        "id": task_id,
        "query": research_query,
        "status": "Initializing",
        "progress": 0,
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "completed": False,
        "results": None,
    }
    
    # Start the research process in a background thread
    def run_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get tools
            tools = get_tools()
            
            # Run the research pipeline
            final_state = loop.run_until_complete(
                run_research_pipeline(
                    task_id=task_id,
                    research_query=research_query,
                    llm=llm,
                    tools=tools,
                    embeddings=embeddings,
                    storage_dir=os.path.join(STORAGE_DIR, "research_tasks"),
                    callback=lambda event, state: update_task_status(task_id, event, state),
                )
            )
            
            # Update task with final results
            tasks[task_id]["completed"] = True
            tasks[task_id]["status"] = "Completed"
            tasks[task_id]["results"] = {
                "final_report": final_state.get("final_report", {}),
                "research_query": final_state.get("research_query", ""),
            }
        except Exception as e:
            tasks[task_id]["status"] = f"Error: {str(e)}"
            print(f"Error in research task {task_id}: {e}")
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/research/<task_id>", methods=["GET"])
def get_research_status(task_id):
    """Get the status of a research task."""
    task = tasks.get(task_id)
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    return jsonify(task)


@app.route("/api/research/<task_id>/results", methods=["GET"])
def get_research_results(task_id):
    """Get the results of a completed research task."""
    task = tasks.get(task_id)
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    if not task["completed"]:
        return jsonify({"error": "Task not completed yet"}), 400
    
    return jsonify(task["results"])


@app.route("/storage/<path:filename>")
def serve_storage(filename):
    """Serve files from the storage directory."""
    return send_from_directory(STORAGE_DIR, filename)


def run_api_server():
    """Run the API server."""
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)


if __name__ == "__main__":
    run_api_server()
