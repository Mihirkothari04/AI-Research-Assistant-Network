"""
Base agent class for the Research Assistant Network.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool


class BaseAgent(ABC):
    """Base class for all agents in the Research Assistant Network."""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: The name of the agent
            description: A description of the agent's role
            llm: The language model to use
            tools: Optional list of tools the agent can use
            memory: Optional memory for the agent
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
    
    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return outputs.
        
        Args:
            inputs: Dictionary of input data
            
        Returns:
            Dictionary of output data
        """
        pass
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.
        
        Returns:
            The system prompt string
        """
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""You are {self.name}, {self.description}
        
Your task is to process the inputs provided and generate appropriate outputs.

You have access to the following tools:
{tools_str}

Always think step by step and explain your reasoning clearly.
"""
