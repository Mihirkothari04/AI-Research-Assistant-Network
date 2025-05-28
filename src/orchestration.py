"""
Orchestration module for the Research Assistant Network.
"""
from typing import Any, Dict, List, Optional, TypedDict, Callable
import os
import uuid
import asyncio

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END

from src.agents.info_gatherer_agent import InfoGathererAgent
from src.agents.insight_generator_agent import InsightGeneratorAgent
from src.agents.data_analyst_agent import DataAnalystAgent
from src.agents.hypothesis_tester_agent import HypothesisTesterAgent
from src.agents.report_compiler_agent import ReportCompilerAgent


class AgentState(TypedDict):
    """Shared state passed between agents."""
    
    research_query: str
    research_materials: List[Any]
    insights: Dict[str, Any]
    structured_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    visualizations: List[str]
    hypotheses: List[Dict[str, Any]]
    experiment_results: List[Dict[str, Any]]
    interpreted_results: List[Dict[str, Any]]
    needs_more_data: bool
    final_report: Dict[str, Any]


def needs_more_data(state: AgentState) -> bool:
    """
    Determine if more data is needed based on the state.
    
    Args:
        state: The current state
        
    Returns:
        Boolean indicating whether more data is needed
    """
    return state.get("needs_more_data", False)


def create_research_graph(
    llm: BaseLLM,
    tools: Dict[str, List[BaseTool]],
    memory: Optional[BaseMemory] = None,
    embeddings: Optional[Embeddings] = None,
    storage_dir: str = "./storage",
):
    """
    Create the directed graph for the research workflow.
    
    Args:
        llm: The language model to use
        tools: Dictionary of tools for each agent
        memory: Optional memory for the agents
        embeddings: Optional embeddings model
        storage_dir: Base directory for storage
        
    Returns:
        Compiled StateGraph
    """
    # Create directories
    os.makedirs(os.path.join(storage_dir, "vector_db"), exist_ok=True)
    os.makedirs(os.path.join(storage_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(storage_dir, "experiments"), exist_ok=True)
    os.makedirs(os.path.join(storage_dir, "reports"), exist_ok=True)
    
    # Initialize agents
    info_gatherer = InfoGathererAgent(
        llm=llm,
        tools=tools.get("info_gatherer", []),
        memory=memory,
        embeddings=embeddings,
        vector_store_path=os.path.join(storage_dir, "vector_db"),
    )
    
    insight_generator = InsightGeneratorAgent(
        llm=llm,
        tools=tools.get("insight_generator", []),
        memory=memory,
    )
    
    data_analyst = DataAnalystAgent(
        llm=llm,
        tools=tools.get("data_analyst", []),
        memory=memory,
        output_dir=os.path.join(storage_dir, "visualizations"),
    )
    
    hypothesis_tester = HypothesisTesterAgent(
        llm=llm,
        tools=tools.get("hypothesis_tester", []),
        memory=memory,
        output_dir=os.path.join(storage_dir, "experiments"),
    )
    
    report_compiler = ReportCompilerAgent(
        llm=llm,
        tools=tools.get("report_compiler", []),
        memory=memory,
        output_dir=os.path.join(storage_dir, "reports"),
    )
    
    # Define the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("info_gatherer", info_gatherer.process)
    workflow.add_node("insight_generator", insight_generator.process)
    workflow.add_node("data_analyst", data_analyst.process)
    workflow.add_node("hypothesis_tester", hypothesis_tester.process)
    workflow.add_node("report_compiler", report_compiler.process)
    
    # Define edges (workflow)
    workflow.add_edge("info_gatherer", "insight_generator")
    workflow.add_edge("insight_generator", "data_analyst")
    workflow.add_edge("data_analyst", "hypothesis_tester")
    
    # Add conditional edges for feedback loops
    workflow.add_conditional_edges(
        "hypothesis_tester",
        needs_more_data,
        {
            True: "info_gatherer",  # Loop back for more data if needed
            False: "report_compiler"  # Continue to report if sufficient
        }
    )
    
    # Final node goes to END
    workflow.add_edge("report_compiler", END)
    
    # Set entry point
    workflow.set_entry_point("info_gatherer")
    
    return workflow.compile()


async def run_research_pipeline(
    task_id: str,
    research_query: str,
    llm: BaseLLM,
    tools: Dict[str, List[BaseTool]],
    memory: Optional[BaseMemory] = None,
    embeddings: Optional[Embeddings] = None,
    storage_dir: str = "./storage",
    callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """
    Run the research pipeline for a given query.
    
    Args:
        task_id: Unique identifier for the task
        research_query: The research query to process
        llm: The language model to use
        tools: Dictionary of tools for each agent
        memory: Optional memory for the agents
        embeddings: Optional embeddings model
        storage_dir: Base directory for storage
        callback: Optional callback function to report progress
        
    Returns:
        Final state with research results
    """
    # Create task-specific storage directory
    task_dir = os.path.join(storage_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Create the research graph
    graph = create_research_graph(
        llm=llm,
        tools=tools,
        memory=memory,
        embeddings=embeddings,
        storage_dir=task_dir,
    )
    
    # Initialize the state
    initial_state = AgentState(
        research_query=research_query,
        research_materials=[],
        insights={},
        structured_data={},
        analysis_results={},
        visualizations=[],
        hypotheses=[],
        experiment_results=[],
        interpreted_results=[],
        needs_more_data=False,
        final_report={},
    )
    
    # Execute the graph
    async for event, state in graph.astream(initial_state):
        # Report progress if callback is provided
        if callback:
            callback(event, state)
    
    return state
