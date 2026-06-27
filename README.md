# langgraph-research-agent

A LangGraph supervisor pipeline: a supervisor node routes between `research`, `writer`, and `qa`
worker nodes, with a human-in-the-loop approval step before QA runs.

## Setup

```bash
uv sync
```

Add an OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-...
```

## Commands

| Command | Description |
| --- | --- |
| `uv run langgraph dev` | Start the LangGraph API server (Studio UI) for the `supervisor_pipeline` graph. |
| `uv run python main.py` | Run the standalone script: streams the pipeline, pauses before QA for approval, then resumes. |
| `uv add <package>` | Add a new dependency to the project. |
| `uv sync` | Install/update dependencies from `pyproject.toml`/`uv.lock`. |
