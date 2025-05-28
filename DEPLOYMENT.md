# Research Assistant Network Deployment Guide

This guide provides instructions for deploying and using the Research Assistant Network, a fully autonomous multi-agent AI research system.

## System Overview

The Research Assistant Network is a team of specialized AI agents that work together to tackle open-ended research queries. The system follows a pipeline approach:

1. **InfoGathererAgent**: Retrieves relevant literature and data from academic and web sources
2. **InsightGeneratorAgent**: Summarizes findings and extracts key insights
3. **DataAnalystAgent**: Processes and visualizes data
4. **HypothesisTesterAgent**: Proposes and tests hypotheses
5. **ReportCompilerAgent**: Compiles all outputs into a coherent report

## Deployment Options

### Option 1: Run Locally with Python

1. Install dependencies:
```bash
cd research_assistant_network
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_api_key" # Optional
export BING_SEARCH_API_KEY="your_bing_search_api_key" # Optional
```

3. Run the application:
```bash
python main.py
```

4. Access the web interface at http://localhost:5000

### Option 2: Deploy with Docker

1. Set environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
BING_SEARCH_API_KEY=your_bing_search_api_key
```

2. Build and run with Docker Compose:
```bash
docker-compose up -d
```

3. Access the web interface at http://localhost:5000

### Option 3: Deploy to Production

For production deployment, we recommend using a cloud provider like AWS, Google Cloud, or Azure. The system can be deployed using:

- Docker containers with Kubernetes
- Cloud Run or AWS Fargate for serverless container deployment
- Virtual machines with Docker installed

## Using the Research Assistant Network

1. Access the web interface at the deployment URL
2. Click "New Research Query" and enter your research question
3. The system will autonomously process your query through all agents
4. Once complete, you can view and download the research report

## System Architecture

The system is built with a modular architecture:

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

## Troubleshooting

If you encounter issues:

1. Check the logs in the `logs` directory
2. Verify API keys are correctly set
3. Ensure all dependencies are installed
4. Check network connectivity for external API access

## Extending the System

The modular design allows for easy extension:

- Add new agent types by extending the BaseAgent class
- Integrate additional data sources in the InfoGathererAgent
- Add new analysis tools to the DataAnalystAgent
- Customize report formats in the ReportCompilerAgent
