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

from pocketflow import Flow
from typing import Dict, Any, List, Optional
from nodes import (
    CoordinatorNode,
    PlannerNode,
    ExecutorNode,
    SimpleExecutorNode,
    MultiAgentAnswerNode,
    log_thinking,
)
from utils.osdk_client import get_osdk_client
from utils.config import AppConfig


def create_multi_agent_flow(config: Optional[AppConfig] = None) -> Flow:
    """
    Create the multi-agent coordinator-planner-executor flow.
    
    Args:
        config: Optional AppConfig for node settings (retries, etc.)
    
    Returns:
        Flow: The configured multi-agent flow
    """
    # Get retry settings from config
    max_retries = config.agent.max_retries if config else 2
    retry_delay = config.agent.retry_delay if config else 1.0
    
    # Create node instances with retry settings
    coordinator = CoordinatorNode(max_retries=max_retries, wait=retry_delay)
    simple_executor = SimpleExecutorNode(max_retries=max_retries, wait=retry_delay)
    planner = PlannerNode(max_retries=max_retries, wait=retry_delay)
    executor = ExecutorNode(max_retries=max_retries, wait=retry_delay)
    answer = MultiAgentAnswerNode(max_retries=max_retries, wait=retry_delay)
    
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


def initialize_multi_agent_state(query: str, config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """
    Initialize the shared state for multi-agent execution.
    
    Args:
        query: The user's question
        config: Optional AppConfig for settings
        
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
        except Exception as e:
            # Schema loading is best-effort, continue without this schema
            pass
    
    # Use config if provided, otherwise use defaults
    if config is None:
        config = get_default_config()
    
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
        "config": config,  # Store config in shared state for nodes to access
    }


def run_multi_agent_query(
    query: str, 
    conversation_history: Optional[List[Dict[str, str]]] = None,
    config: Optional[AppConfig] = None
) -> Dict[str, Any]:
    """
    Run a query through the multi-agent flow.
    
    Args:
        query: The user's question
        conversation_history: Optional list of previous messages
        config: Optional AppConfig for settings
        
    Returns:
        Dictionary with results
    """
    shared = initialize_multi_agent_state(query, config)
    
    if conversation_history:
        shared["messages"] = conversation_history
    
    flow = create_multi_agent_flow(config)
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
