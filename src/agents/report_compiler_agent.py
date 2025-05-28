"""
ReportCompilerAgent for compiling the final research report.
"""
from typing import Any, Dict, List, Optional
import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent


class ReportCompilerAgent(BaseAgent):
    """Agent responsible for compiling the final research report."""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        output_dir: str = "./storage/reports",
    ):
        """
        Initialize the ReportCompilerAgent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools the agent can use
            memory: Optional memory for the agent
            output_dir: Directory to save reports
        """
        super().__init__(
            name="ReportCompilerAgent",
            description="an expert at compiling research findings into comprehensive reports",
            llm=llm,
            tools=tools,
            memory=memory,
        )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all research outputs to compile a final report.
        
        Args:
            inputs: Dictionary containing all research outputs
            
        Returns:
            Dictionary with report file paths
        """
        research_query = inputs.get("research_query", "")
        overall_summary = inputs.get("overall_summary", "")
        key_insights = inputs.get("key_insights", [])
        group_summaries = inputs.get("group_summaries", {})
        analysis_results = inputs.get("analysis_results", {})
        visualizations = inputs.get("visualizations", [])
        hypotheses = inputs.get("hypotheses", [])
        interpreted_results = inputs.get("interpreted_results", [])
        
        if not research_query:
            raise ValueError("Research query is required")
        
        # Step 1: Generate report outline
        report_outline = await self._generate_report_outline(research_query, key_insights, interpreted_results)
        
        # Step 2: Create Jupyter notebook
        notebook_path = await self._create_jupyter_notebook(
            research_query,
            report_outline,
            overall_summary,
            key_insights,
            group_summaries,
            analysis_results,
            visualizations,
            hypotheses,
            interpreted_results,
        )
        
        # Step 3: Generate PDF report (optional)
        pdf_path = await self._generate_pdf_report(notebook_path)
        
        # Step 4: Create slide deck (optional)
        slides_path = await self._create_slide_deck(
            research_query,
            report_outline,
            overall_summary,
            key_insights,
            visualizations,
            interpreted_results,
        )
        
        return {
            "research_query": research_query,
            "report_outline": report_outline,
            "notebook_path": notebook_path,
            "pdf_path": pdf_path,
            "slides_path": slides_path,
        }
    
    async def _generate_report_outline(
        self,
        research_query: str,
        key_insights: List[Dict[str, Any]],
        interpreted_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate an outline for the research report.
        
        Args:
            research_query: The original research query
            key_insights: List of key insights
            interpreted_results: List of interpreted experiment results
            
        Returns:
            List of report sections
        """
        # In a real implementation, this would use the LLM to generate an outline
        # For now, we'll create a placeholder outline
        
        outline = [
            {
                "title": "Executive Summary",
                "content": "Summary of the research findings and key insights.",
            },
            {
                "title": "Introduction",
                "content": f"Background and context for the research query: {research_query}",
            },
            {
                "title": "Methodology",
                "content": "Description of the research methodology, including data sources and analysis techniques.",
            },
            {
                "title": "Literature Review",
                "content": "Summary of the relevant literature and previous research.",
            },
            {
                "title": "Data Analysis",
                "content": "Presentation and analysis of the data collected.",
            },
            {
                "title": "Findings",
                "content": "Detailed presentation of the research findings.",
            },
            {
                "title": "Discussion",
                "content": "Interpretation and discussion of the findings in relation to the research query.",
            },
            {
                "title": "Conclusion",
                "content": "Summary of the main findings and their implications.",
            },
            {
                "title": "References",
                "content": "List of references cited in the report.",
            },
        ]
        
        return outline
    
    async def _create_jupyter_notebook(
        self,
        research_query: str,
        report_outline: List[Dict[str, Any]],
        overall_summary: str,
        key_insights: List[Dict[str, Any]],
        group_summaries: Dict[str, str],
        analysis_results: Dict[str, Any],
        visualizations: List[str],
        hypotheses: List[Dict[str, Any]],
        interpreted_results: List[Dict[str, Any]],
    ) -> str:
        """
        Create a Jupyter notebook with the research report.
        
        Args:
            Various research outputs and report components
            
        Returns:
            Path to the created notebook
        """
        # Create a new notebook
        notebook = new_notebook()
        
        # Add title and introduction
        notebook.cells.append(new_markdown_cell(f"# Research Report: {research_query}"))
        notebook.cells.append(new_markdown_cell("## Executive Summary"))
        notebook.cells.append(new_markdown_cell(overall_summary))
        
        # Add sections based on the outline
        for section in report_outline:
            if section["title"] == "Executive Summary":
                continue  # Already added
            
            notebook.cells.append(new_markdown_cell(f"## {section['title']}"))
            notebook.cells.append(new_markdown_cell(section["content"]))
            
            # Add specific content based on the section
            if section["title"] == "Introduction":
                notebook.cells.append(new_markdown_cell(f"### Research Query\n\n{research_query}"))
            
            elif section["title"] == "Literature Review":
                notebook.cells.append(new_markdown_cell("### Summary of Literature"))
                for source, summary in group_summaries.items():
                    notebook.cells.append(new_markdown_cell(f"#### {source}\n\n{summary}"))
            
            elif section["title"] == "Data Analysis":
                notebook.cells.append(new_markdown_cell("### Analysis Results"))
                
                # Add code cell for displaying analysis results
                code = """
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Display analysis results
analysis_results = {analysis_results}
display(Markdown("#### Descriptive Statistics"))
if 'descriptive_statistics' in analysis_results:
    stats_df = pd.DataFrame([analysis_results['descriptive_statistics']])
    display(stats_df)
""".format(analysis_results=json.dumps(analysis_results))
                
                notebook.cells.append(new_code_cell(code))
                
                # Add visualizations
                notebook.cells.append(new_markdown_cell("### Visualizations"))
                for viz_path in visualizations:
                    viz_filename = os.path.basename(viz_path)
                    notebook.cells.append(new_markdown_cell(f"![{viz_filename}]({viz_path})"))
            
            elif section["title"] == "Findings":
                notebook.cells.append(new_markdown_cell("### Key Insights"))
                for insight in key_insights:
                    notebook.cells.append(new_markdown_cell(f"- {insight.get('insight', '')}"))
                
                notebook.cells.append(new_markdown_cell("### Hypotheses and Results"))
                for result in interpreted_results:
                    status = "Supported" if result.get("supported", False) else "Not Supported"
                    notebook.cells.append(new_markdown_cell(
                        f"**Hypothesis {result.get('hypothesis_id', '')}**: {result.get('hypothesis_statement', '')}\n\n"
                        f"**Status**: {status} (Confidence: {result.get('confidence', 'unknown')})\n\n"
                        f"**Explanation**: {result.get('explanation', '')}\n\n"
                        f"**Implications**: {result.get('implications', '')}\n\n"
                        f"**Limitations**: {result.get('limitations', '')}"
                    ))
        
        # Save the notebook
        task_id = "task_" + research_query.replace(" ", "_")[:20]
        notebook_path = os.path.join(self.output_dir, f"{task_id}_report.ipynb")
        
        with open(notebook_path, "w") as f:
            nbformat.write(notebook, f)
        
        return notebook_path
    
    async def _generate_pdf_report(self, notebook_path: str) -> Optional[str]:
        """
        Generate a PDF report from the Jupyter notebook.
        
        Args:
            notebook_path: Path to the Jupyter notebook
            
        Returns:
            Path to the generated PDF, or None if generation failed
        """
        # In a real implementation, this would use nbconvert to generate a PDF
        # For now, we'll create a placeholder PDF path
        
        pdf_path = notebook_path.replace(".ipynb", ".pdf")
        
        # Simulate PDF generation (in reality, would use nbconvert)
        try:
            # This is a placeholder - in a real implementation, we would use nbconvert
            with open(pdf_path, "w") as f:
                f.write("Placeholder PDF content")
            
            return pdf_path
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
    
    async def _create_slide_deck(
        self,
        research_query: str,
        report_outline: List[Dict[str, Any]],
        overall_summary: str,
        key_insights: List[Dict[str, Any]],
        visualizations: List[str],
        interpreted_results: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Create a slide deck from the research report.
        
        Args:
            Various research outputs and report components
            
        Returns:
            Path to the created slide deck, or None if creation failed
        """
        # In a real implementation, this would use a library like Marp or reveal.js
        # For now, we'll create a placeholder HTML file
        
        task_id = "task_" + research_query.replace(" ", "_")[:20]
        slides_path = os.path.join(self.output_dir, f"{task_id}_slides.html")
        
        try:
            # Create a simple HTML slide deck
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {research_query}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        .slide {{
            width: 100%;
            height: 100vh;
            padding: 2em;
            box-sizing: border-box;
            page-break-after: always;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 0.5em;
        }}
        h2 {{
            font-size: 2em;
            margin-bottom: 0.5em;
        }}
        p {{
            font-size: 1.5em;
            line-height: 1.4;
        }}
        ul {{
            font-size: 1.5em;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="slide">
        <h1>Research Report</h1>
        <h2>{research_query}</h2>
    </div>
    
    <div class="slide">
        <h1>Executive Summary</h1>
        <p>{overall_summary}</p>
    </div>
    
    <div class="slide">
        <h1>Key Insights</h1>
        <ul>
"""
            
            # Add key insights
            for insight in key_insights:
                html_content += f"            <li>{insight.get('insight', '')}</li>\n"
            
            html_content += """
        </ul>
    </div>
    
    <div class="slide">
        <h1>Methodology</h1>
        <p>This research was conducted using a multi-agent AI system that autonomously gathered information, analyzed data, tested hypotheses, and compiled results.</p>
    </div>
"""
            
            # Add findings slides
            html_content += """
    <div class="slide">
        <h1>Findings</h1>
        <p>Summary of key findings from the research:</p>
        <ul>
"""
            
            # Add interpreted results
            for result in interpreted_results:
                status = "Supported" if result.get("supported", False) else "Not Supported"
                html_content += f"            <li><strong>{result.get('hypothesis_id', '')}</strong>: {result.get('hypothesis_statement', '')} - {status}</li>\n"
            
            html_content += """
        </ul>
    </div>
    
    <div class="slide">
        <h1>Conclusion</h1>
        <p>This research provides valuable insights into the query and suggests several directions for future investigation.</p>
    </div>
</body>
</html>
"""
            
            with open(slides_path, "w") as f:
                f.write(html_content)
            
            return slides_path
        except Exception as e:
            print(f"Error creating slide deck: {e}")
            return None
