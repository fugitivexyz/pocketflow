"""
Multi-agent flow for the Palantir OSDK chatbot.

This flow implements a coordinator-planner-executor pattern:
1. Coordinator: Classifies query complexity
2. For simple queries: Direct execution
3. For complex queries: Planner creates step-by-step plan
4. Executor: Runs each step in the plan
5. Answer: Generates final response

```mermaid
flowchart TB
    Coordinator[Coordinator] -->|simple| SimpleExec[Simple Executor]
    Coordinator -->|complex| Planner[Planner]
    
    Planner -->|plan ready| Executor[Executor]
    Executor -->|continue| Executor
    Executor -->|complete| Answer[Answer]
    
    SimpleExec -->|done| Answer
    
    subgraph "Complex Query Handling"
        Planner
        Executor
    end
```

This multi-agent approach is useful for:
- Complex queries requiring multiple data fetches
- Queries that need cross-object analysis
- Multi-dimensional comparisons
"""

from pocketflow import Flow, Node
from typing import Dict, Any, List, Optional
from nodes import (
    CoordinatorNode,
    PlannerNode,
    ExecutorNode,
    log_thinking,
)
from utils.osdk_client import get_osdk_client
from utils.call_llm import call_llm, extract_yaml
from prompts import SIMPLE_EXECUTOR_PROMPT, MULTI_AGENT_ANSWER_PROMPT


class SimpleExecutorNode(Node):
    """
    Simple executor for straightforward queries.
    
    Handles basic queries that don't need multi-step planning:
    - List objects of a type
    - Filter by simple criteria
    - Basic lookups
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for simple execution."""
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what simple action to take."""
        prompt = SIMPLE_EXECUTOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        response = call_llm(prompt)
        return {"prompt": prompt, "response": response}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Execute the simple query."""
        log_thinking(shared, "⚡ Simple Execution", exec_res["response"])
        
        try:
            query_spec = extract_yaml(exec_res["response"])
            
            client = get_osdk_client()
            df = client.query_objects(
                object_type=query_spec.get("object_type", ""),
                filters=query_spec.get("filters", {}),
                limit=query_spec.get("limit", 100),
            )
            
            shared["fetched_data"] = df
            log_thinking(
                shared,
                "📊 Data Retrieved",
                f"Fetched {len(df)} rows from {query_spec.get('object_type')}"
            )
            
        except Exception as e:
            log_thinking(shared, "⚠️ Execution Error", str(e))
            shared["error"] = str(e)
        
        return "answer"


class MultiAgentAnswerNode(Node):
    """
    Answer node for multi-agent flow.
    
    Synthesizes results from potentially multiple data sources
    and plan executions.
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Gather all accumulated data and context."""
        return {
            "query": shared.get("current_query", ""),
            "fetched_data": shared.get("fetched_data"),
            "accumulated_data": shared.get("accumulated_data", {}),
            "execution_plan": shared.get("execution_plan", []),
            "query_complexity": shared.get("query_complexity", "simple"),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive answer."""
        # Build context from all data sources
        data_context = []
        
        if prep_res["fetched_data"] is not None:
            df = prep_res["fetched_data"]
            data_context.append(f"Main data: {len(df)} rows, columns: {list(df.columns)}")
            data_context.append(f"Sample:\n{df.head(5).to_string()}")
        
        for key, df in prep_res["accumulated_data"].items():
            if df is not None and not df.empty:
                data_context.append(f"\n{key}: {len(df)} rows")
        
        data_str = "\n".join(data_context) if data_context else "No data retrieved."
        
        plan_str = "No execution plan (simple query)."
        if prep_res["execution_plan"]:
            plan_str = f"Executed {len(prep_res['execution_plan'])} planned steps."
        
        prompt = MULTI_AGENT_ANSWER_PROMPT.format(
            query=prep_res["query"],
            query_complexity=prep_res["query_complexity"],
            plan_str=plan_str,
            data_str=data_str
        )
        
        response = call_llm(prompt)
        return {"answer": response, "prompt": prompt}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store final answer."""
        shared["final_answer"] = exec_res["answer"]
        log_thinking(shared, "💬 Final Answer", exec_res["answer"])
        return "done"


def create_multi_agent_flow() -> Flow:
    """
    Create the multi-agent coordinator-planner-executor flow.
    
    Returns:
        Flow: The configured multi-agent flow
    """
    # Create node instances
    coordinator = CoordinatorNode()
    simple_executor = SimpleExecutorNode()
    planner = PlannerNode()
    executor = ExecutorNode()
    answer = MultiAgentAnswerNode()
    
    # Connect coordinator to handlers
    coordinator - "execute_simple" >> simple_executor
    coordinator - "plan" >> planner
    
    # Simple path goes directly to answer
    simple_executor - "answer" >> answer
    
    # Complex path: plan -> execute -> answer
    planner - "execute_plan" >> executor
    executor - "continue" >> executor  # Loop for multiple steps
    executor - "answer" >> answer
    
    # Create and return the flow
    return Flow(start=coordinator)


def initialize_multi_agent_state(query: str) -> Dict[str, Any]:
    """
    Initialize the shared state for multi-agent execution.
    
    Args:
        query: The user's question
        
    Returns:
        Initialized shared state dictionary
    """
    client = get_osdk_client()
    object_types = client.list_object_types()
    
    # Pre-load schemas for better planning
    schemas = {}
    for obj_type in object_types:
        try:
            schemas[obj_type] = client.get_object_schema(obj_type)
        except:
            pass
    
    return {
        "current_query": query,
        "messages": [],
        "object_types": object_types,
        "schemas": schemas,
        "fetched_data": None,
        "accumulated_data": {},
        "execution_plan": [],
        "current_step": 0,
        "query_complexity": None,
        "final_answer": None,
        "thinking_steps": [],
        "action_history": [],  # Track actions for loop detection
    }


def run_multi_agent_query(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Run a query through the multi-agent flow.
    
    Args:
        query: The user's question
        conversation_history: Optional list of previous messages
        
    Returns:
        Dictionary with results
    """
    shared = initialize_multi_agent_state(query)
    
    if conversation_history:
        shared["messages"] = conversation_history
    
    flow = create_multi_agent_flow()
    flow.run(shared)
    
    return {
        "final_answer": shared.get("final_answer", "I couldn't generate a response."),
        "fetched_data": shared.get("fetched_data"),
        "accumulated_data": shared.get("accumulated_data", {}),
        "figure": shared.get("figure"),
        "thinking_steps": shared.get("thinking_steps", []),
        "query_complexity": shared.get("query_complexity"),
        "execution_plan": shared.get("execution_plan", []),
    }


# Test the multi-agent flow
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Agent Flow")
    print("=" * 60)
    
    # Test with a complex query
    test_query = "Compare turbidity results across different surfactants and concentrations"
    print(f"\nQuery: {test_query}\n")
    
    result = run_multi_agent_query(test_query)
    
    print("\n" + "=" * 60)
    print(f"QUERY COMPLEXITY: {result['query_complexity']}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("THINKING STEPS:")
    print("=" * 60)
    for step in result["thinking_steps"]:
        print(f"\n{step['type']}")
        print("-" * 40)
        content = step["content"]
        print(content[:500] if len(content) > 500 else content)
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])
