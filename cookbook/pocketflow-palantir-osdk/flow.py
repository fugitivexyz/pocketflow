"""
Multi-agent flow for the Palantir OSDK chatbot.

Implements Coordinator-Planner-Executor pattern:
1. Coordinator: Classifies query complexity
2. Simple queries: Direct execution
3. Complex queries: Planner creates step-by-step plan
4. Executor: Runs each step
5. Answer: Generates final response

```mermaid
flowchart TB
    Coordinator -->|simple| SimpleExec[Simple Executor]
    Coordinator -->|complex| Planner
    
    Planner --> Executor
    Executor -->|continue| Executor
    Executor -->|complete| Answer
    
    SimpleExec --> Answer
```
"""

import threading
from typing import Dict, Any, List, Optional

from pocketflow import Flow
from nodes import CoordinatorNode, PlannerNode, ExecutorNode, SimpleExecutorNode, AnswerNode
from utils.osdk_client import get_osdk_client
from utils.config import Config, load_config, get_default_config


def create_flow(config: Optional[Config] = None) -> Flow:
    """Create the multi-agent coordinator-planner-executor flow."""
    max_retries = config.max_retries if config else 2
    retry_delay = config.retry_delay if config else 1.0
    
    coordinator = CoordinatorNode(max_retries=max_retries, wait=retry_delay)
    simple_executor = SimpleExecutorNode(max_retries=max_retries, wait=retry_delay)
    planner = PlannerNode(max_retries=max_retries, wait=retry_delay)
    executor = ExecutorNode(max_retries=max_retries, wait=retry_delay)
    answer = AnswerNode(max_retries=max_retries, wait=retry_delay)
    
    # Connect flow
    coordinator - "execute_simple" >> simple_executor
    coordinator - "plan" >> planner
    simple_executor - "answer" >> answer
    planner - "execute_plan" >> executor
    executor - "continue" >> executor
    executor - "answer" >> answer
    
    return Flow(start=coordinator)


def initialize_state(
    query: str,
    config: Optional[Config] = None,
    stop_event: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """Initialize shared state for flow execution."""
    client = get_osdk_client()
    object_types = client.list_object_types()
    
    # Pre-load schemas
    schemas = {}
    for obj_type in object_types:
        try:
            schemas[obj_type] = client.get_object_schema(obj_type)
        except Exception:
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
        "config": config or get_default_config(),
        "stop_event": stop_event,
    }


def run_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    config: Optional[Config] = None,
    stop_event: Optional[threading.Event] = None
) -> Dict[str, Any]:
    """
    Run a query through the multi-agent flow.
    
    Args:
        query: The user's question
        conversation_history: Optional list of previous messages
        config: Optional Config for settings
        stop_event: Optional threading.Event for cancellation
        
    Returns:
        Dictionary with results
    """
    shared = initialize_state(query, config, stop_event)
    
    if conversation_history:
        shared["messages"] = conversation_history
    
    flow = create_flow(config)
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


# Backward compatibility alias
run_multi_agent_query = run_query
create_multi_agent_flow = create_flow
initialize_multi_agent_state = initialize_state


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Agent Flow")
    print("=" * 60)
    
    test_query = "Compare turbidity results across different surfactants"
    print(f"\nQuery: {test_query}\n")
    
    result = run_query(test_query)
    
    print(f"\nComplexity: {result['query_complexity']}")
    print("\nThinking steps:")
    for step in result["thinking_steps"]:
        print(f"  {step['type']}: {step['content'][:100]}...")
    
    print("\nFinal Answer:")
    print(result["final_answer"])
