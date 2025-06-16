# Agent

A modern, async, and scalable agent orchestration framework leveraging state-of-the-art language models like **Claude** and **GPT-4** for natural language understanding, generation, and reasoning.

## Features

- **Async/Await**: All agent and service methods are async for better performance.
- **Caching**: In-memory caching for expensive operations like embeddings, keyword extraction, search results, and LLM completions.
- **Rate Limiting**: In-memory rate limiting for API calls to LLM, search, and knowledge base operations.
- **Error Handling & Retries**: Robust error handling and retry logic for all network/API calls.
- **Monitoring**: Prometheus metrics, structured logging, and OpenTelemetry tracing for detailed execution flow.
- **Cost Tracking**: Token and cost tracking for every LLM and search API call, with aggregated cost reporting.
- **All configuration lives in `config.yaml`**: All configuration lives in **`config.yaml`** (no `config.toml`).  You can override any setting via environment variables – see `src/config.py` for the mapping logic.
- **Code is executed directly on your machine**: Code is executed **directly on your machine** by `TerminalRunner` – Docker or Firejail is **not** required.  Use your own shell utilities and system packages.

## Project Structure

```
.
├── agent/                  # Core agent logic
│   ├── core/               # Core agent modules (orchestrator, knowledge base)
├── src/                    # Source code
│   ├── agents/             # Agent implementations
│   ├── bert/               # BERT/KeyBERT for keyword extraction
│   ├── browser/            # Web search and browser automation
│   ├── llm/                # LLM client and inference
│   ├── services/           # Service integrations
│   └── utils/              # Utility functions
├── data/                   # Data storage (cache, logs, projects)
├── docs/                   # Documentation
├── tests/                  # Unit and integration tests
├── config.yaml             # Configuration file
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set up a virtual environment:**
   ```sh
   # On Windows (PowerShell)
   python -m venv venv
   venv\Scripts\Activate.ps1

   # On Windows (cmd)
   python -m venv venv
   venv\Scripts\activate.bat

   # On Unix/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys and configuration values.

## Running the Project

### Backend

1. **Activate the virtual environment** (if not already activated):
   ```sh
   venv\Scripts\Activate.ps1   # PowerShell
   # or
   venv\Scripts\activate.bat   # cmd
   ```

2. **Run the main entry point:**
   ```sh
   python main.py
   ```

## Configuration

- **API Keys**: Store all API keys in `.env`.
- **Configurable Values**: All configurable values are in `config.yaml`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Swarm framework
- Qdrant for vector storage
- All contributors and users

## Monitoring & Cost Tracking

- **Prometheus Metrics**: Available at `http://localhost:9090/metrics`.
- **Token & Cost Tracking**: Monitor usage in the logs.

## Real Example Usage

### Knowledge Base

```python
from agent.core.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
doc_id = kb.add_document(
    text="Python is a high-level programming language.",
    metadata={"title": "Python Doc", "category": "Programming"}
)
results = kb.search("high-level programming language", limit=1)
print(results)
```

### Orchestrator

```python
from agent.core.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
result = orchestrator.run(
    start_agent="planner",
    messages=[{"role": "user", "content": "Build a FastAPI web server."}],
    context_variables={"research_needed": True, "code_generation_needed": True},
    max_turns=6
)
print(result)
```

### Agent Manager

```python
from agent.core.agents import AgentManager

manager = AgentManager()
result = manager.run(
    agent_name="planner",
    messages=[{"role": "user", "content": "Create a project plan."}],
    context_variables={"research_needed": True}
)
print(result)
```

### Project Manager

```python
from src.project import ProjectManager

pm = ProjectManager()
pm.create_project("Test Project")
pm.add_message_from_user("Test Project", "Hello, this is a test message.")
messages = pm.get_messages("Test Project")
print(messages)
```

### State Manager

```python
from src.state import AgentState

sm = AgentState()
sm.create_state("Test Project")
new_state = sm.new_state()
new_state["internal_monologue"] = "Working on the first step."
sm.add_to_current_state("Test Project", new_state)
state_stack = sm.get_current_state("Test Project")
print(state_stack)
```

### Logger

```python
from src.logger import Logger

logger = Logger()
logger.info("This is an info message.")
logger.error("This is an error message.")
```

Install Microsoft Visual C++ Build Tools:
   Download and install the Microsoft C++ Build Tools

   During installation, select "Desktop development with C++" workload
   
   Make sure to include the Windows 10 SDK and MSVC v143 - VS 2022 C++ x64/x86 build tools

https://visualstudio.microsoft.com/visual-cpp-build-tools/