"""
DataAnalystAgent for processing and visualizing data.
"""
from typing import Any, Dict, List, Optional
import os
import json
import tempfile

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain.tools.python.tool import PythonREPLTool

from .base_agent import BaseAgent


class DataAnalystAgent(BaseAgent):
    """Agent responsible for data processing and visualization."""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        output_dir: str = "./storage/visualizations",
    ):
        """
        Initialize the DataAnalystAgent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools the agent can use
            memory: Optional memory for the agent
            output_dir: Directory to save visualizations
        """
        # Ensure Python REPL tool is included
        if tools is None:
            tools = []
        
        python_repl_tool = next((t for t in tools if isinstance(t, PythonREPLTool)), None)
        if python_repl_tool is None:
            tools.append(PythonREPLTool())
        
        super().__init__(
            name="DataAnalystAgent",
            description="an expert at processing data and creating visualizations",
            llm=llm,
            tools=tools,
            memory=memory,
        )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process insights and research materials to generate data analysis and visualizations.
        
        Args:
            inputs: Dictionary containing research_query, insights, and research_materials
            
        Returns:
            Dictionary with processed data and visualizations
        """
        research_query = inputs.get("research_query", "")
        insights = inputs.get("key_insights", [])
        research_materials = inputs.get("research_materials", [])
        
        if not research_query:
            raise ValueError("Research query is required")
        
        # Step 1: Extract structured data from research materials
        structured_data = await self._extract_structured_data(research_materials)
        
        # Step 2: Generate analysis plan based on research query and insights
        analysis_plan = await self._generate_analysis_plan(research_query, insights, structured_data)
        
        # Step 3: Execute analysis and generate visualizations
        analysis_results, visualizations = await self._execute_analysis(analysis_plan, structured_data)
        
        # Step 4: Interpret the analysis results
        interpretation = await self._interpret_results(analysis_results, research_query)
        
        return {
            "research_query": research_query,
            "structured_data": structured_data,
            "analysis_plan": analysis_plan,
            "analysis_results": analysis_results,
            "visualizations": visualizations,
            "interpretation": interpretation,
        }
    
    async def _extract_structured_data(self, materials: List[Document]) -> Dict[str, Any]:
        """
        Extract structured data from research materials.
        
        Args:
            materials: List of Document objects
            
        Returns:
            Dictionary with structured data
        """
        # In a real implementation, this would use the LLM to extract tables, numbers, etc.
        # For now, we'll create a placeholder structured data object
        
        structured_data = {
            "tables": [],
            "numerical_data": {},
            "time_series": {},
            "categorical_data": {},
        }
        
        # Extract tables and numerical data from documents
        for doc in materials:
            content = doc.page_content
            
            # Simple example: look for patterns that might indicate tables or numerical data
            # This is a very simplified approach - in reality, we would use more sophisticated methods
            
            # Check for potential table markers
            if "| " in content and " |" in content:
                # This might be a markdown table
                structured_data["tables"].append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": "Sample table content (placeholder)",
                    "format": "markdown"
                })
            
            # Check for numerical patterns (very simplified)
            if any(c.isdigit() for c in content):
                structured_data["numerical_data"]["sample_data"] = [1, 2, 3, 4, 5]  # Placeholder
        
        return structured_data
    
    async def _generate_analysis_plan(
        self, research_query: str, insights: List[Dict[str, Any]], structured_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate a plan for data analysis based on the research query and available data.
        
        Args:
            research_query: The original research query
            insights: List of insights from the InsightGeneratorAgent
            structured_data: Dictionary with structured data
            
        Returns:
            List of analysis tasks to perform
        """
        # In a real implementation, this would use the LLM to generate an analysis plan
        # For now, we'll create a placeholder analysis plan
        
        analysis_plan = [
            {
                "type": "descriptive_statistics",
                "description": "Calculate basic statistics for numerical data",
                "data_source": "numerical_data",
            },
            {
                "type": "visualization",
                "description": "Create a bar chart of sample data",
                "data_source": "numerical_data.sample_data",
                "visualization_type": "bar_chart",
                "output_file": os.path.join(self.output_dir, "sample_bar_chart.png"),
            },
            {
                "type": "visualization",
                "description": "Create a line chart of sample data",
                "data_source": "numerical_data.sample_data",
                "visualization_type": "line_chart",
                "output_file": os.path.join(self.output_dir, "sample_line_chart.png"),
            }
        ]
        
        return analysis_plan
    
    async def _execute_analysis(
        self, analysis_plan: List[Dict[str, Any]], structured_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Execute the analysis plan and generate visualizations.
        
        Args:
            analysis_plan: List of analysis tasks to perform
            structured_data: Dictionary with structured data
            
        Returns:
            Tuple of (analysis results, list of visualization file paths)
        """
        analysis_results = {}
        visualizations = []
        
        # Find the Python REPL tool
        python_tool = next((t for t in self.tools if isinstance(t, PythonREPLTool)), None)
        if not python_tool:
            raise ValueError("Python REPL tool is required for analysis")
        
        for task in analysis_plan:
            task_type = task.get("type", "")
            
            if task_type == "descriptive_statistics":
                # Generate descriptive statistics
                code = """
import numpy as np
import pandas as pd

# Sample data for demonstration
data = [1, 2, 3, 4, 5]

# Calculate statistics
stats = {
    'mean': np.mean(data),
    'median': np.median(data),
    'std': np.std(data),
    'min': np.min(data),
    'max': np.max(data)
}

stats
"""
                try:
                    result = python_tool.invoke({"code": code})
                    analysis_results["descriptive_statistics"] = eval(result)
                except Exception as e:
                    print(f"Error executing descriptive statistics: {e}")
                    analysis_results["descriptive_statistics"] = {"error": str(e)}
            
            elif task_type == "visualization":
                # Generate visualization
                viz_type = task.get("visualization_type", "")
                output_file = task.get("output_file", "")
                
                if viz_type == "bar_chart":
                    code = f"""
import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
data = [1, 2, 3, 4, 5]
labels = ['A', 'B', 'C', 'D', 'E']

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, data)
plt.title('Sample Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('{output_file}')
plt.close()

'{output_file}'
"""
                elif viz_type == "line_chart":
                    code = f"""
import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
x = np.arange(5)
y = [1, 2, 3, 4, 5]

# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o')
plt.title('Sample Line Chart')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('{output_file}')
plt.close()

'{output_file}'
"""
                else:
                    continue
                
                try:
                    result = python_tool.invoke({"code": code})
                    if os.path.exists(output_file):
                        visualizations.append(output_file)
                except Exception as e:
                    print(f"Error generating visualization: {e}")
        
        return analysis_results, visualizations
    
    async def _interpret_results(
        self, analysis_results: Dict[str, Any], research_query: str
    ) -> str:
        """
        Interpret the analysis results in the context of the research query.
        
        Args:
            analysis_results: Dictionary with analysis results
            research_query: The original research query
            
        Returns:
            Interpretation text
        """
        # In a real implementation, this would use the LLM to interpret the results
        # For now, we'll create a placeholder interpretation
        
        return """
This is a placeholder interpretation of the analysis results. In a real implementation, 
the LLM would generate a detailed interpretation based on the analysis results and how 
they relate to the original research query.

The interpretation would include:
1. Key findings from the data analysis
2. How these findings relate to the research query
3. Any patterns or trends identified
4. Potential implications of the findings
5. Suggestions for further analysis
"""
