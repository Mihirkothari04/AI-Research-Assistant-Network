"""
InsightGeneratorAgent for summarizing findings and extracting insights.
"""
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from .base_agent import BaseAgent


class InsightGeneratorAgent(BaseAgent):
    """Agent responsible for summarizing findings and extracting key insights."""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the InsightGeneratorAgent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools the agent can use
            memory: Optional memory for the agent
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between document chunks
        """
        super().__init__(
            name="InsightGeneratorAgent",
            description="an expert at summarizing research materials and extracting key insights",
            llm=llm,
            tools=tools,
            memory=memory,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.summarize_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            verbose=True,
        )
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research materials and generate summaries and insights.
        
        Args:
            inputs: Dictionary containing research_query and research_materials
            
        Returns:
            Dictionary with summaries and insights
        """
        research_query = inputs.get("research_query", "")
        research_materials = inputs.get("research_materials", [])
        
        if not research_query:
            raise ValueError("Research query is required")
        
        if not research_materials:
            raise ValueError("Research materials are required")
        
        # Step 1: Group materials by source or topic
        grouped_materials = self._group_materials(research_materials)
        
        # Step 2: Generate summaries for each group
        group_summaries = await self._generate_group_summaries(grouped_materials, research_query)
        
        # Step 3: Extract key insights across all materials
        key_insights = await self._extract_key_insights(research_materials, research_query)
        
        # Step 4: Generate an overall summary
        overall_summary = await self._generate_overall_summary(group_summaries, research_query)
        
        return {
            "research_query": research_query,
            "group_summaries": group_summaries,
            "key_insights": key_insights,
            "overall_summary": overall_summary,
        }
    
    def _group_materials(self, materials: List[Document]) -> Dict[str, List[Document]]:
        """
        Group research materials by source or topic.
        
        Args:
            materials: List of Document objects
            
        Returns:
            Dictionary with groups of documents
        """
        grouped = {}
        
        for doc in materials:
            source = doc.metadata.get("source", "unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(doc)
        
        return grouped
    
    async def _generate_group_summaries(
        self, grouped_materials: Dict[str, List[Document]], research_query: str
    ) -> Dict[str, str]:
        """
        Generate summaries for each group of materials.
        
        Args:
            grouped_materials: Dictionary with groups of documents
            research_query: The original research query
            
        Returns:
            Dictionary with summaries for each group
        """
        summaries = {}
        
        for source, docs in grouped_materials.items():
            if not docs:
                continue
                
            # Split documents if they are too large
            all_splits = []
            for doc in docs:
                splits = self.text_splitter.split_text(doc.page_content)
                all_splits.extend([Document(page_content=s, metadata=doc.metadata) for s in splits])
            
            # Generate summary using the summarize chain
            if all_splits:
                try:
                    summary = self.summarize_chain.run(
                        input_documents=all_splits,
                        question=f"Summarize these documents in relation to the research query: {research_query}"
                    )
                    summaries[source] = summary
                except Exception as e:
                    print(f"Error generating summary for {source}: {e}")
                    summaries[source] = "Error generating summary."
        
        return summaries
    
    async def _extract_key_insights(
        self, materials: List[Document], research_query: str
    ) -> List[Dict[str, Any]]:
        """
        Extract key insights from all research materials.
        
        Args:
            materials: List of Document objects
            research_query: The original research query
            
        Returns:
            List of key insights with metadata
        """
        # Combine all document content
        combined_text = "\n\n".join([doc.page_content for doc in materials])
        
        # Split the combined text
        splits = self.text_splitter.split_text(combined_text)
        split_docs = [Document(page_content=s) for s in splits]
        
        # Use the LLM to extract insights
        insights = []
        
        # This is a simplified version - in a real implementation, we would use the LLM
        # For now, we'll create a placeholder insight
        insights.append({
            "insight": "This is a placeholder insight. In a real implementation, the LLM would generate insights based on the research materials.",
            "relevance": "high",
            "sources": ["placeholder"],
        })
        
        return insights
    
    async def _generate_overall_summary(
        self, group_summaries: Dict[str, str], research_query: str
    ) -> str:
        """
        Generate an overall summary based on group summaries.
        
        Args:
            group_summaries: Dictionary with summaries for each group
            research_query: The original research query
            
        Returns:
            Overall summary text
        """
        # Combine all group summaries
        combined_summary = "\n\n".join([f"{source}:\n{summary}" for source, summary in group_summaries.items()])
        
        # Use the LLM to generate an overall summary
        prompt = f"""
        Based on the following summaries from different sources, generate a comprehensive overall summary
        that addresses the research query: "{research_query}"
        
        Summaries:
        {combined_summary}
        
        Your overall summary should:
        1. Synthesize information across all sources
        2. Highlight key findings and consensus views
        3. Note any contradictions or gaps in the research
        4. Relate everything back to the original research query
        """
        
        # This is a simplified version - in a real implementation, we would use the LLM
        # For now, we'll return a placeholder summary
        return "This is a placeholder overall summary. In a real implementation, the LLM would generate a comprehensive summary based on all the research materials."
