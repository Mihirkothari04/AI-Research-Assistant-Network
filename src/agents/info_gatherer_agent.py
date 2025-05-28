"""
InfoGathererAgent for retrieving relevant literature and data.
"""
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .base_agent import BaseAgent


class InfoGathererAgent(BaseAgent):
    """Agent responsible for retrieving relevant literature and data."""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        memory: Optional[BaseMemory] = None,
        embeddings: Optional[Embeddings] = None,
        vector_store_path: str = "./storage/vector_db",
    ):
        """
        Initialize the InfoGathererAgent.
        
        Args:
            llm: The language model to use
            tools: List of search and retrieval tools
            memory: Optional memory for the agent
            embeddings: Embeddings model for vector storage
            vector_store_path: Path to store vector database
        """
        super().__init__(
            name="InfoGathererAgent",
            description="an expert at finding and retrieving relevant academic papers, datasets, and information from various sources",
            llm=llm,
            tools=tools,
            memory=memory,
        )
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.vector_store = None
        
        # Initialize vector store if embeddings are provided
        if embeddings:
            self.vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings,
            )
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research query and retrieve relevant information.
        
        Args:
            inputs: Dictionary containing at least a 'research_query' key
            
        Returns:
            Dictionary with retrieved research materials
        """
        research_query = inputs.get("research_query", "")
        if not research_query:
            raise ValueError("Research query is required")
        
        # Step 1: Analyze the research query to identify key topics and search terms
        search_plan = await self._create_search_plan(research_query)
        
        # Step 2: Execute searches using available tools
        search_results = await self._execute_searches(search_plan)
        
        # Step 3: Process and filter the search results
        processed_materials = await self._process_search_results(search_results, research_query)
        
        # Step 4: Index the processed materials in the vector store
        if self.vector_store and self.embeddings:
            await self._index_materials(processed_materials)
        
        return {
            "research_query": research_query,
            "search_plan": search_plan,
            "research_materials": processed_materials,
        }
    
    async def _create_search_plan(self, research_query: str) -> Dict[str, Any]:
        """
        Create a plan for searching based on the research query.
        
        Args:
            research_query: The research query to analyze
            
        Returns:
            A search plan with key topics and search strategies
        """
        # Use the LLM to analyze the query and create a search plan
        prompt = f"""
        Analyze the following research query and create a comprehensive search plan:
        
        Research Query: {research_query}
        
        Your search plan should include:
        1. Key topics and concepts to search for
        2. Relevant academic databases to query (e.g., Semantic Scholar, arXiv)
        3. Web search queries to find supplementary information
        4. Any specific datasets that might be relevant
        5. Search priority (which topics or sources to focus on first)
        
        Format your response as a structured JSON object.
        """
        
        # This is a simplified version - in a real implementation, we would use the LLM to generate this
        # For now, we'll create a basic search plan manually
        return {
            "key_topics": [research_query],
            "academic_sources": ["semantic_scholar", "arxiv"],
            "web_sources": ["general_search"],
            "dataset_sources": [],
            "search_priority": "academic_first"
        }
    
    async def _execute_searches(self, search_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute searches based on the search plan using available tools.
        
        Args:
            search_plan: The search plan to execute
            
        Returns:
            List of search results
        """
        results = []
        
        # Execute academic searches
        for source in search_plan["academic_sources"]:
            for topic in search_plan["key_topics"]:
                # Find the appropriate tool for this source
                tool = next((t for t in self.tools if source in t.name.lower()), None)
                if tool:
                    try:
                        # Execute the tool with the topic as input
                        result = tool.invoke({"query": topic, "limit": 10})
                        results.append({
                            "source": source,
                            "topic": topic,
                            "results": result
                        })
                    except Exception as e:
                        print(f"Error executing {source} search for {topic}: {e}")
        
        # Execute web searches
        for source in search_plan["web_sources"]:
            for topic in search_plan["key_topics"]:
                tool = next((t for t in self.tools if source in t.name.lower()), None)
                if tool:
                    try:
                        result = tool.invoke({"query": topic, "limit": 10})
                        results.append({
                            "source": source,
                            "topic": topic,
                            "results": result
                        })
                    except Exception as e:
                        print(f"Error executing {source} search for {topic}: {e}")
        
        return results
    
    async def _process_search_results(
        self, search_results: List[Dict[str, Any]], research_query: str
    ) -> List[Document]:
        """
        Process and filter search results to create a list of relevant documents.
        
        Args:
            search_results: List of search results from different sources
            research_query: The original research query
            
        Returns:
            List of processed Document objects
        """
        processed_documents = []
        
        for result_group in search_results:
            source = result_group["source"]
            results = result_group["results"]
            
            # Process results based on their source
            if isinstance(results, list):
                for item in results:
                    # Convert each result to a Document object
                    if isinstance(item, dict):
                        # Extract relevant fields based on source
                        title = item.get("title", "")
                        content = item.get("abstract", item.get("content", ""))
                        url = item.get("url", item.get("link", ""))
                        authors = item.get("authors", [])
                        
                        # Create metadata
                        metadata = {
                            "source": source,
                            "title": title,
                            "url": url,
                            "authors": authors if isinstance(authors, list) else [authors],
                        }
                        
                        # Create Document object
                        doc = Document(page_content=content, metadata=metadata)
                        processed_documents.append(doc)
        
        return processed_documents
    
    async def _index_materials(self, materials: List[Document]) -> None:
        """
        Index the processed materials in the vector store.
        
        Args:
            materials: List of Document objects to index
        """
        if not self.vector_store:
            return
        
        try:
            self.vector_store.add_documents(materials)
            self.vector_store.persist()
        except Exception as e:
            print(f"Error indexing materials: {e}")
