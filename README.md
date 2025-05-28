# Research Assistant Network

A fully-autonomous multi-agent AI research system that tackles open-ended research queries.

## Overview

The Research Assistant Network is a team of specialized AI agents that work together to autonomously conduct literature review, analysis, hypothesis testing, and reporting. The system follows a pipeline approach with five specialized agents:

1. **InfoGathererAgent**: Retrieves relevant literature and data from academic and web sources
2. **InsightGeneratorAgent**: Summarizes findings and extracts key insights
3. **DataAnalystAgent**: Processes and visualizes data
4. **HypothesisTesterAgent**: Proposes and tests hypotheses
5. **ReportCompilerAgent**: Compiles all outputs into a coherent report

## Features

- Fully autonomous operation with zero human-in-the-loop
- Modular architecture for easy extension and customization
- Integration with academic APIs (Semantic Scholar, arXiv, CrossRef)
- Data visualization and analysis capabilities
- Hypothesis generation and testing
- Comprehensive report generation (Jupyter notebooks, PDFs, slides)
- Web interface for submitting research queries and viewing results

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended for deployment)
- OpenAI API key

### Option 1: Docker Installation (Recommended)

1. Clone or extract this repository
2. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key  # Optional
   BING_SEARCH_API_KEY=your_bing_search_api_key  # Optional
   ```
3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```
4. Access the web interface at http://localhost:5000

### Option 2: Local Installation

1. Clone or extract this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set environment variables:
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY="your_openai_api_key"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your_openai_api_key
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_openai_api_key"
   ```
5. Run the application:
   ```bash
   python main.py
   ```
6. Access the web interface at http://localhost:5000

## Usage

### Submitting a Research Query

1. Access the web interface at http://localhost:5000
2. Click "New Research Query" and enter your research question
3. The system will autonomously process your query through all agents
4. Once complete, you can view and download the research report

### Example Queries

- "What are the latest advancements in quantum computing?"
- "How does climate change affect biodiversity in coral reefs?"
- "What are the ethical implications of large language models?"
- "What is the current state of research on fusion energy?"

## System Architecture

The system uses LangChain with LangGraph as the primary orchestration framework, which provides a directed graph structure for the agent pipeline. The architecture is modular and follows a clean separation of concerns:

- **Agent Framework**: LangChain with LangGraph for orchestration
- **Language Models**: OpenAI GPT-4 for agent reasoning
- **Vector Database**: ChromaDB for document storage and retrieval
- **API Layer**: Flask for web interface and REST API
- **Containerization**: Docker for deployment

## Configuration

Configuration options are available in `src/config/config.py`. Key settings include:

- API keys for various services
- LLM model selection
- API server settings
- Storage directories
- Logging configuration

## Project Structure

```
research_assistant_network/
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── info_gatherer_agent.py
│   │   ├── insight_generator_agent.py
│   │   ├── data_analyst_agent.py
│   │   ├── hypothesis_tester_agent.py
│   │   └── report_compiler_agent.py
│   ├── api/
│   │   ├── api.py
│   │   └── templates/
│   │       └── index.html
│   ├── config/
│   │   └── config.py
│   ├── utils/
│   └── orchestration.py
├── storage/
│   ├── research_tasks/
│   ├── vector_db/
│   └── reports/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
└── README.md
```

## Extending the System

The modular design allows for easy extension:

- Add new agent types by extending the BaseAgent class
- Integrate additional data sources in the InfoGathererAgent
- Add new analysis tools to the DataAnalystAgent
- Customize report formats in the ReportCompilerAgent

## Troubleshooting

If you encounter issues:

1. Check the logs in the `logs` directory
2. Verify API keys are correctly set
3. Ensure all dependencies are installed
4. Check network connectivity for external API access

## License

This project is available for your use and modification.

## Acknowledgements

This system was designed based on research in multi-agent AI systems, including:
- LangChain and LangGraph frameworks
- OpenAI's language models
- Academic research on autonomous AI systems
