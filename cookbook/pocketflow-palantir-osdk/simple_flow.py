"""
Simple single-node flow for basic OSDK queries.

This demonstrates when NOT to use the multi-agent pattern.
Use this for:
- Simple lookups (e.g., "Show all experiments")
- Basic filtering (e.g., "List completed experiments")
- Direct data retrieval without complex analysis

Use multi-agent flow (flow.py) for:
- Complex multi-step queries
- Cross-object analysis
- Visualization requests
- Queries requiring planning

```mermaid
flowchart LR
    Query[QueryNode] --> Answer
```
"""

from pocketflow import Node, Flow
from typing import Dict, Any, Optional

from utils.call_llm import call_llm, extract_yaml
from utils.osdk_client import get_osdk_client


SIMPLE_QUERY_PROMPT = """You are a data query assistant. Determine what to query.

## USER QUERY
{query}

## AVAILABLE OBJECT TYPES
{object_types}

## TASK
Return the query parameters in YAML:
```yaml
object_type: <type to query>
filters: {{}}
limit: 100
```
"""

SIMPLE_ANSWER_PROMPT = """Answer the user's question based on the data.

## USER QUERY
{query}

## DATA RETRIEVED ({count} rows)
Columns: {columns}
Sample:
{sample}

## TASK
Provide a clear, helpful answer.
"""


class SimpleQueryNode(Node):
    """Single node that queries data and generates an answer."""
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        client = get_osdk_client()
        return {
            "query": shared.get("query", ""),
            "object_types": client.list_object_types(),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Determine what to query
        query_prompt = SIMPLE_QUERY_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        query_response = call_llm(query_prompt)
        query_spec = extract_yaml(query_response)
        
        # Step 2: Execute query
        client = get_osdk_client()
        df = client.query_objects(
            object_type=query_spec.get("object_type", ""),
            filters=query_spec.get("filters", {}),
            limit=query_spec.get("limit", 100),
        )
        
        # Step 3: Generate answer
        answer_prompt = SIMPLE_ANSWER_PROMPT.format(
            query=prep_res["query"],
            count=len(df),
            columns=list(df.columns),
            sample=df.head(10).to_string() if not df.empty else "No data found"
        )
        answer = call_llm(answer_prompt)
        
        return {"data": df, "answer": answer, "query_spec": query_spec}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        shared["data"] = exec_res["data"]
        shared["answer"] = exec_res["answer"]
        return "done"


def create_simple_flow() -> Flow:
    """Create the simple single-node flow."""
    return Flow(start=SimpleQueryNode(max_retries=2, wait=1))


def run_simple_query(query: str) -> Dict[str, Any]:
    """Run a simple query through the single-node flow."""
    shared = {"query": query}
    flow = create_simple_flow()
    flow.run(shared)
    return {
        "answer": shared.get("answer", "No answer generated."),
        "data": shared.get("data"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Flow Example")
    print("=" * 60)
    print("Use this for basic queries. Use flow.py for complex analysis.\n")
    
    # Example simple queries
    queries = [
        "Show me all experiments",
        "List completed experiments",
        "What proteins are available?",
    ]
    
    for query in queries[:1]:  # Just test one
        print(f"Query: {query}")
        result = run_simple_query(query)
        print(f"Answer: {result['answer']}\n")
        if result['data'] is not None:
            print(f"Data shape: {result['data'].shape}")
