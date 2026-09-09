# AI Article Search Chatbot

A stateful AI-powered article search chatbot that uses **LLM tool calling, Tavily web search, Newspaper3k, and LangGraph** to discover, extract, evaluate, and refine relevant web articles.

## Key Features

* **LLM Tool Calling** — Uses a `web_search` tool with automatic tool selection.
* **Web Search** — Retrieves relevant articles using Tavily.
* **Full Article Extraction** — Extracts article content from URLs using Newspaper3k.
* **Relevance Filtering** — Uses an LLM to evaluate and score articles according to user requirements.
* **Structured Output** — Returns results as a strict JSON array.
* **Stateful Refinement** — Allows users to update and refine previous results through follow-up requests.
* **LangGraph Workflow** — Organizes the application into reusable states and processing nodes.
* **Docker Support** — Includes Docker configuration for containerized execution.

## Architecture

```text
User Input
    │
    ▼
Prepare Input & Prompt
    │
    ▼
First LLM Call
    │
    ├── Tool Call ──► Tavily Web Search
    │                       │
    │                       ▼
    │                Newspaper3k
    │                Article Extraction
    │                       │
    └───────────────────────┘
                            │
                            ▼
                    Second LLM Call
                            │
                            ▼
                  Relevance Filtering
                            │
                            ▼
                    JSON Results
                            │
                            ▼
                     User Updates
                            │
                            ▼
                  Re-run Filtering
```

## Technology Stack

* Python
* LangGraph
* LangChain
* LLM / Generative AI
* Tavily
* Newspaper3k
* Docker

## Project Structure

```text
article-search-chatbot/
│
├── graph.py              # LangGraph workflow
├── llm.py                # LLM configuration
├── main.py               # Application entry point
├── nodes.py              # Workflow nodes
├── retriever.py          # Search and article retrieval
├── state.py              # Shared application state
├── utils.py              # Utility functions
│
├── Dockerfile
├── Docker-compose.yml
├── requirements.txt
└── README.md
```

## Output Format

The chatbot returns a JSON array containing:

```json
[
  {
    "title": "Article Title",
    "url": "https://example.com/article",
    "relevance_score": 0.92,
    "full_content": "Complete article content...",
    "suggested_topic": "Artificial Intelligence"
  }
]
```

## Stateful Interaction

The chatbot supports follow-up instructions after the initial search.

**Example:**

```text
User:
Only keep articles with a relevance score above 0.85.

Chatbot:
Returns updated filtered results.
```

This allows the filtering stage to be called again using updated user requirements while maintaining the workflow state.

## Setup

### Clone

```bash
git clone https://github.com/HAJIRASHAHID/article-search-chatbot.git
cd article-search-chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
LLM_API_KEY=your_llm_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Run

```bash
python main.py
```

## Project Purpose

This project demonstrates the practical implementation of **AI agents, LLM tool calling, web-enabled retrieval, structured generation, and stateful workflows** using LangGraph.

## Author

**Hajira Shahid**

GitHub: [HAJIRASHAHID](https://github.com/HAJIRASHAHID)
