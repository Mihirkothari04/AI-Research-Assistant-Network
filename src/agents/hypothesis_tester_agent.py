"""
HypothesisTesterAgent for proposing and testing hypotheses.
"""
from typing import Any, Dict, List, Optional
import os

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain.tools.python.tool import PythonREPLTool

from .base_agent import BaseAgent


class HypothesisTesterAgent(BaseAgent):
    """Agent responsible for proposing and testing hypotheses."""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        output_dir: str = "./storage/experiments",
    ):
        """
        Initialize the HypothesisTesterAgent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools the agent can use
            memory: Optional memory for the agent
            output_dir: Directory to save experiment results
        """
        # Ensure Python REPL tool is included
        if tools is None:
            tools = []
        
        python_repl_tool = next((t for t in tools if isinstance(t, PythonREPLTool)), None)
        if python_repl_tool is None:
            tools.append(PythonREPLTool())
        
        super().__init__(
            name="HypothesisTesterAgent",
            description="an expert at proposing and testing hypotheses based on research findings",
            llm=llm,
            tools=tools,
            memory=memory,
        )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process insights and analysis results to generate and test hypotheses.
        
        Args:
            inputs: Dictionary containing research_query, insights, and analysis_results
            
        Returns:
            Dictionary with hypotheses and test results
        """
        research_query = inputs.get("research_query", "")
        insights = inputs.get("key_insights", [])
        analysis_results = inputs.get("analysis_results", {})
        structured_data = inputs.get("structured_data", {})
        
        if not research_query:
            raise ValueError("Research query is required")
        
        # Step 1: Generate hypotheses based on insights and analysis
        hypotheses = await self._generate_hypotheses(research_query, insights, analysis_results)
        
        # Step 2: Design experiments to test the hypotheses
        experiments = await self._design_experiments(hypotheses, structured_data)
        
        # Step 3: Execute the experiments
        experiment_results = await self._execute_experiments(experiments)
        
        # Step 4: Analyze and interpret the experiment results
        interpreted_results = await self._interpret_results(hypotheses, experiment_results, research_query)
        
        # Step 5: Determine if additional data is needed
        needs_more_data = await self._evaluate_data_needs(interpreted_results)
        
        return {
            "research_query": research_query,
            "hypotheses": hypotheses,
            "experiments": experiments,
            "experiment_results": experiment_results,
            "interpreted_results": interpreted_results,
            "needs_more_data": needs_more_data,
        }
    
    async def _generate_hypotheses(
        self, research_query: str, insights: List[Dict[str, Any]], analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on insights and analysis results.
        
        Args:
            research_query: The original research query
            insights: List of insights from the InsightGeneratorAgent
            analysis_results: Dictionary with analysis results
            
        Returns:
            List of hypotheses
        """
        # In a real implementation, this would use the LLM to generate hypotheses
        # For now, we'll create placeholder hypotheses
        
        hypotheses = [
            {
                "id": "H1",
                "statement": "This is a placeholder hypothesis based on the research query.",
                "rationale": "This hypothesis is derived from the key insights and analysis results.",
                "testability": "high",
                "related_insights": ["insight1"],
            },
            {
                "id": "H2",
                "statement": "This is another placeholder hypothesis exploring an alternative explanation.",
                "rationale": "This hypothesis considers a different perspective on the research question.",
                "testability": "medium",
                "related_insights": ["insight1"],
            }
        ]
        
        return hypotheses
    
    async def _design_experiments(
        self, hypotheses: List[Dict[str, Any]], structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Design experiments to test the hypotheses.
        
        Args:
            hypotheses: List of hypotheses
            structured_data: Dictionary with structured data
            
        Returns:
            List of experiment designs
        """
        # In a real implementation, this would use the LLM to design experiments
        # For now, we'll create placeholder experiment designs
        
        experiments = []
        
        for i, hypothesis in enumerate(hypotheses):
            experiment = {
                "id": f"E{i+1}",
                "hypothesis_id": hypothesis["id"],
                "title": f"Experiment for {hypothesis['id']}",
                "description": f"This experiment tests {hypothesis['statement']}",
                "method": "simulation",
                "code_template": """
import numpy as np
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(42)
data = np.random.normal(0, 1, 100)

# Analyze data
mean = np.mean(data)
std = np.std(data)
p_value = np.mean(data > 0)

# Visualize results
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20)
plt.axvline(mean, color='r', linestyle='--')
plt.title('Experiment Results')
plt.savefig('{output_file}')
plt.close()

# Return results
results = {{
    'mean': mean,
    'std': std,
    'p_value': p_value,
    'significant': p_value < 0.05
}}

results
""",
                "output_file": os.path.join(self.output_dir, f"experiment_{i+1}_results.png"),
            }
            
            experiments.append(experiment)
        
        return experiments
    
    async def _execute_experiments(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the designed experiments.
        
        Args:
            experiments: List of experiment designs
            
        Returns:
            List of experiment results
        """
        experiment_results = []
        
        # Find the Python REPL tool
        python_tool = next((t for t in self.tools if isinstance(t, PythonREPLTool)), None)
        if not python_tool:
            raise ValueError("Python REPL tool is required for experiments")
        
        for experiment in experiments:
            # Replace placeholders in the code template
            code = experiment["code_template"].format(output_file=experiment["output_file"])
            
            try:
                # Execute the experiment code
                result = python_tool.invoke({"code": code})
                
                # Parse the results
                parsed_result = eval(result)
                
                # Add to experiment results
                experiment_results.append({
                    "experiment_id": experiment["id"],
                    "hypothesis_id": experiment["hypothesis_id"],
                    "results": parsed_result,
                    "visualization": experiment["output_file"] if os.path.exists(experiment["output_file"]) else None,
                    "success": True,
                })
            except Exception as e:
                print(f"Error executing experiment {experiment['id']}: {e}")
                experiment_results.append({
                    "experiment_id": experiment["id"],
                    "hypothesis_id": experiment["hypothesis_id"],
                    "error": str(e),
                    "success": False,
                })
        
        return experiment_results
    
    async def _interpret_results(
        self, 
        hypotheses: List[Dict[str, Any]], 
        experiment_results: List[Dict[str, Any]],
        research_query: str
    ) -> List[Dict[str, Any]]:
        """
        Interpret the experiment results in relation to the hypotheses.
        
        Args:
            hypotheses: List of hypotheses
            experiment_results: List of experiment results
            research_query: The original research query
            
        Returns:
            List of interpreted results
        """
        # In a real implementation, this would use the LLM to interpret the results
        # For now, we'll create placeholder interpretations
        
        interpreted_results = []
        
        for hypothesis in hypotheses:
            # Find the corresponding experiment results
            results = [r for r in experiment_results if r["hypothesis_id"] == hypothesis["id"]]
            
            if not results:
                continue
            
            # Create interpretation
            interpretation = {
                "hypothesis_id": hypothesis["id"],
                "hypothesis_statement": hypothesis["statement"],
                "supported": any(r.get("results", {}).get("significant", False) for r in results if r["success"]),
                "confidence": "medium",
                "explanation": f"This is a placeholder interpretation of the results for hypothesis {hypothesis['id']}. In a real implementation, the LLM would generate a detailed interpretation based on the experiment results.",
                "implications": "These findings suggest further research is needed in this area.",
                "limitations": "The simulated nature of the experiment limits the generalizability of these findings.",
            }
            
            interpreted_results.append(interpretation)
        
        return interpreted_results
    
    async def _evaluate_data_needs(self, interpreted_results: List[Dict[str, Any]]) -> bool:
        """
        Evaluate whether additional data is needed based on the interpreted results.
        
        Args:
            interpreted_results: List of interpreted results
            
        Returns:
            Boolean indicating whether more data is needed
        """
        # In a real implementation, this would use the LLM to evaluate data needs
        # For now, we'll return a fixed value
        
        # Check if any hypotheses have low confidence or are not supported
        low_confidence = any(r["confidence"] == "low" for r in interpreted_results)
        not_supported = any(not r["supported"] for r in interpreted_results)
        
        # For demonstration purposes, we'll say we don't need more data
        return False
