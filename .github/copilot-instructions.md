# PocketFlow AI Coding Instructions

> **Philosophy**: PocketFlow is a 100-line minimalist LLM framework. Simplicity is paramount—avoid complexity, dependencies, and vendor lock-in.

## Architecture Overview

PocketFlow models LLM workflows as **Graph + Shared Store**:

- **Core** (`pocketflow/__init__.py`): ~100 lines defining `Node`, `Flow`, `BatchNode`, `AsyncNode`, and variants
- **Shared Store**: In-memory dict for inter-node communication (passed to `run()`)
- **Transitions**: Default (`>>`) or conditional (`node - "action" >> other_node`)

```
Node.prep(shared) → Node.exec(prep_res) → Node.post(shared, prep_res, exec_res) → returns action string
```

## Project Structure Pattern

All cookbook examples and new projects follow this structure:

```
project/
├── main.py          # Entry point: creates shared dict, runs flow
├── nodes.py         # Node class definitions (prep/exec/post methods)
├── flow.py          # Flow construction: creates & connects nodes
├── utils/           # External API wrappers (LLM, search, embeddings)
│   └── call_llm.py  # Each util has if __name__ == "__main__" test
└── docs/design.md   # High-level design (required before implementation)
```

## Node Implementation Pattern

```python
from pocketflow import Node

class MyNode(Node):
    def prep(self, shared):
        return shared["input_key"]  # Read from shared store

    def exec(self, prep_res):
        return call_llm(prep_res)   # Do work (LLM, API calls)

    def post(self, shared, prep_res, exec_res):
        shared["output_key"] = exec_res  # Write to shared store
        return "action_name"  # or None for default transition
```

## Flow Connection Patterns

```python
# Sequential (default transition)
node1 >> node2 >> node3

# Conditional branching (based on post() return value)
decide - "search" >> search_node
decide - "answer" >> answer_node

# Loop back
search_node - "decide" >> decide
```

## Key Design Patterns

| Pattern        | When to Use                        | Example Location                   |
| -------------- | ---------------------------------- | ---------------------------------- |
| **Agent**      | Autonomous decisions with tool use | `cookbook/pocketflow-agent/`       |
| **Workflow**   | Linear multi-step pipelines        | `cookbook/pocketflow-workflow/`    |
| **RAG**        | Retrieval + generation             | `cookbook/pocketflow-rag/`         |
| **BatchNode**  | Process lists of items             | `cookbook/pocketflow-batch/`       |
| **AsyncNode**  | I/O-bound parallel operations      | `cookbook/pocketflow-async-basic/` |
| **Map-Reduce** | Split → process → combine          | `cookbook/pocketflow-map-reduce/`  |

## Critical Conventions

1. **No exception handling in utils**: Let Node's retry mechanism handle failures

   ```python
   class MyNode(Node):
       def __init__(self):
           super().__init__(max_retries=3, wait=1)  # Built-in retry
   ```

2. **Shared store is the data contract**: All nodes read/write via `shared` dict

   ```python
   shared = {"input": data, "results": {}}  # Initialize before flow.run()
   ```

3. **Post returns action strings**: For conditional flow, return the edge label

   ```python
   def post(self, shared, prep_res, exec_res):
       return "success" if valid else "retry"  # Must match flow edges
   ```

4. **Utilities are external interfaces only**:
   - ✅ API calls (LLM, web search, database)
   - ❌ LLM logic (summarization, analysis) — that goes in nodes

## Running Tests

```bash
cd /path/to/PocketFlow
python -m pytest tests/ -v
```

## Common Cookbook Commands

Most cookbook examples run with:

```bash
cd cookbook/pocketflow-<example>
pip install -r requirements.txt
python main.py
```

## Agentic Coding Workflow

When building new LLM systems:

1. **Design first** → Create `docs/design.md` with flow diagram
2. **Utilities second** → Implement & test external API wrappers
3. **Nodes third** → Define prep/exec/post for each step
4. **Flow last** → Connect nodes with transitions
5. **Iterate** → Expect 100+ iterations on steps 2-4

> See `.cursorrules` or `docs/guide.md` for the complete 8-step agentic coding process.
