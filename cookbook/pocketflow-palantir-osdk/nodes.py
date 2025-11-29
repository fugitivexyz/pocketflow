"""
PocketFlow nodes for the Palantir OSDK chatbot.

This module contains all multi-agent nodes:
- CoordinatorNode: Routes queries to simple or complex handlers
- PlannerNode: Creates execution plans for complex queries
- ExecutorNode: Executes individual plan steps
- SimpleExecutorNode: Handles simple single-step queries
- MultiAgentAnswerNode: Generates final answers

All nodes log their prompts and responses to shared["thinking_steps"] for UI transparency.
"""

from pocketflow import Node, Flow
import yaml
import pandas as pd
import threading
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from utils.call_llm import call_llm, extract_yaml
from utils.osdk_client import get_osdk_client
from utils.visualization import generate_chart
from utils.streaming import log_thinking_streaming
from prompts import COORDINATOR_PROMPT, PLANNER_PROMPT, SIMPLE_EXECUTOR_PROMPT, MULTI_AGENT_ANSWER_PROMPT

if TYPE_CHECKING:
    from utils.config import AppConfig


def get_config(shared: Dict[str, Any]) -> Optional["AppConfig"]:
    """Get config from shared store, returns None if not present."""
    return shared.get("config")


def log_thinking(shared: Dict[str, Any], step_type: str, content: str, level: str = "medium"):
    """
    Log a thinking step for UI transparency.
    
    This function now supports real-time streaming to the UI
    via the streaming callback mechanism.
    
    Args:
        shared: The shared state dictionary
        step_type: Display label for the step (e.g., "🎯 Coordinator", "📊 LLM Response")
        content: The content to display
        level: Verbosity level required to show this step ("low", "medium", "high")
               - "low": Always shown (critical steps only)
               - "medium": Shown at medium and high verbosity
               - "high": Only shown at high verbosity (detailed debug info)
    """
    # Check verbosity level from config
    config = get_config(shared)
    if config:
        verbosity = config.transparency.verbosity_level
        verbosity_order = {"low": 1, "medium": 2, "high": 3}
        # Only log if current verbosity level is >= required level
        current_level = verbosity_order.get(verbosity, 2)
        required_level = verbosity_order.get(level, 2)
        if current_level < required_level:
            return  # Skip this log
    
    log_thinking_streaming(shared, step_type, content)


class CancellationError(Exception):
    """Raised when execution is cancelled by user."""
    pass


def check_cancellation(shared: Dict[str, Any]):
    """Check if cancellation has been requested."""
    stop_event = shared.get("stop_event")
    if stop_event and stop_event.is_set():
        raise CancellationError("Execution cancelled by user")


# =============================================================================
# Multi-Agent Nodes
# =============================================================================

class CoordinatorNode(Node):
    """
    Coordinator for multi-agent system.
    
    Decides whether to:
    - Handle simple queries directly
    - Dispatch to Planner for complex multi-step queries
    
    Config options:
    - agent.complexity_mode: "auto" | "simple" | "complex"
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Get the query and context."""
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "conversation_history": shared.get("messages", []),
            "config": get_config(shared),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Decide if query is simple or complex."""
        config = prep_res.get("config")
        
        # Check for forced complexity mode
        complexity_mode = None
        if config:
            complexity_mode = config.agent.complexity_mode
        
        if complexity_mode == "simple":
            return {"prompt": "[Forced Simple Mode]", "response": "complexity: simple", "forced": True}
        elif complexity_mode == "complex":
            return {"prompt": "[Forced Complex Mode]", "response": "complexity: complex", "forced": True}
        
        # Auto mode - use LLM to decide
        prompt = COORDINATOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        
        response = call_llm(prompt, config=config)
        return {"prompt": prompt, "response": response, "forced": False}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Route to appropriate handler."""
        if exec_res.get("forced"):
            log_thinking(shared, "🎯 Coordinator", f"[Mode Override] {exec_res['response']}", level="medium")
        else:
            log_thinking(shared, "🎯 Coordinator", exec_res["response"], level="high")
        
        try:
            decision = extract_yaml(exec_res["response"])
            complexity = decision.get("complexity", "simple")
        except Exception as e:
            log_thinking(shared, "⚠️ Parse Warning", f"Could not parse complexity decision: {e}, defaulting to simple", level="low")
            complexity = "simple"
        
        shared["query_complexity"] = complexity
        
        if complexity == "complex":
            return "plan"
        else:
            return "execute_simple"


class PlannerNode(Node):
    """
    Planner for complex multi-step queries.
    
    Breaks down complex queries into a sequence of steps.
    
    Config options:
    - agent.max_plan_steps: Maximum number of steps allowed in a plan
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Get query and available schemas."""
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "schemas": shared.get("schemas", {}),
            "config": get_config(shared),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan."""
        config = prep_res.get("config")
        schema_str = yaml.dump(prep_res["schemas"], default_flow_style=False) if prep_res["schemas"] else "No schemas loaded"
        
        prompt = PLANNER_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"],
            schema_str=schema_str
        )
        
        response = call_llm(prompt, config=config)
        return {"prompt": prompt, "response": response, "config": config}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store the plan and proceed to execution."""
        log_thinking(shared, "📝 Plan Created", exec_res["response"], level="medium")
        
        try:
            plan = extract_yaml(exec_res["response"])
            steps = plan.get("plan", [])
            
            # Apply max_plan_steps limit from config
            config = exec_res.get("config")
            if config and len(steps) > config.agent.max_plan_steps:
                log_thinking(shared, "⚠️ Plan Truncated", 
                    f"Plan had {len(steps)} steps, limited to {config.agent.max_plan_steps}", level="low")
                steps = steps[:config.agent.max_plan_steps]
            
            shared["execution_plan"] = steps
            shared["current_step"] = 0
        except Exception as e:
            log_thinking(shared, "⚠️ Planning Error", str(e), level="low")
            shared["execution_plan"] = []
        
        return "execute_plan"


class ExecutorNode(Node):
    """
    Executor for multi-agent system.
    
    Executes individual steps from the plan.
    
    Config options:
    - agent.max_query_results: Limit for fetch operations
    - agent.max_plan_steps: Safety limit for execution loops
    """
    
    def prep(self, shared):
        """Get current step from plan."""
        check_cancellation(shared)
        plan = shared.get("execution_plan", [])
        current_step = shared.get("current_step", 0)
        config = get_config(shared)
        
        # Safety check: limit total steps
        max_steps = 10  # default
        if config:
            max_steps = config.agent.max_plan_steps
        
        if current_step >= len(plan) or current_step >= max_steps:
            return None  # Plan complete or safety limit reached

        return {
            "step": plan[current_step],
            "step_num": current_step,
            "total_steps": len(plan),
            "accumulated_data": shared.get("accumulated_data", {}),
            "fetched_data": shared.get("fetched_data"),  # Current fetched data
            "user_query": shared.get("current_query", ""),  # For dynamic chart inference
            "config": config,
        }

    def exec(self, prep_res):
        """Execute the current step."""
        if prep_res is None:
            return {"action": "complete", "result": None}

        step = prep_res["step"]
        action = step.get("action", "")
        client = get_osdk_client()
        config = prep_res.get("config")

        result = {"action": action, "step": step}

        if action == "fetch":
            # Apply max_query_results limit from config
            limit = step.get("limit", 100)
            if config:
                limit = min(limit, config.agent.max_query_results)
            
            df = client.query_objects(
                object_type=step.get("object_type", ""),
                filters=step.get("filters", {}),
                limit=limit,
            )
            result["data"] = df
            result["success"] = True

        elif action == "analyze":
            # Analysis is handled by the AnalyzeDataNode pattern
            result["instructions"] = step.get("instructions", "")
            result["success"] = True

        elif action == "visualize":
            # Generate visualization with dynamic column inference
            from utils.visualization import validate_chart_spec, infer_chart_spec
            chart_spec = step.get("chart_spec", {})
            user_query = prep_res.get("user_query", "")

            # Get the DataFrame to visualize
            df = prep_res.get("fetched_data")
            if df is None:
                # Try to get from accumulated_data
                acc_data = prep_res.get("accumulated_data", {})
                if acc_data:
                    last_key = list(acc_data.keys())[-1]
                    df = acc_data[last_key]

            if df is not None and not df.empty:
                try:
                    # Use LLM to infer chart spec if x or y are missing
                    if not chart_spec.get("x") or not chart_spec.get("y"):
                        validated_spec = infer_chart_spec(df, user_query, chart_spec)
                    else:
                        validated_spec = validate_chart_spec(chart_spec, df)

                    # Get chart template from config
                    template = "plotly_white"
                    if config and config.visualization.color_theme:
                        template = config.visualization.color_theme
                    
                    fig = generate_chart(df, validated_spec, template=template)
                    result["figure"] = fig
                    result["chart_spec"] = validated_spec
                    result["success"] = True
                except Exception as e:
                    result["error"] = str(e)
                    result["success"] = False
            else:
                result["error"] = "No data available for visualization"
                result["success"] = False

        elif action == "answer":
            result["success"] = True

        return result
    
    def post(self, shared, prep_res, exec_res):
        """Store step result and determine next action."""
        if exec_res["action"] == "complete":
            return "answer"
        
        step = exec_res.get("step", {})
        log_thinking(
            shared,
            f"⚡ Step {prep_res['step_num'] + 1}/{prep_res['total_steps']}: {exec_res['action']}",
            f"Description: {step.get('description', 'N/A')}",
            level="medium"
        )
        
        # Store results
        if exec_res["action"] == "fetch" and "data" in exec_res:
            if "accumulated_data" not in shared:
                shared["accumulated_data"] = {}
            shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
            shared["fetched_data"] = exec_res["data"]  # Also set as current

        # Store visualization figure if generated
        if exec_res["action"] == "visualize":
            if exec_res.get("figure"):
                shared["figure"] = exec_res["figure"]
                log_thinking(shared, "📊 Visualization Created",
                    f"Chart spec: {exec_res.get('chart_spec', {})}", level="medium")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Visualization Error", exec_res["error"], level="low")

        # Advance to next step
        shared["current_step"] = prep_res["step_num"] + 1
        
        # Check if more steps
        if shared["current_step"] >= prep_res["total_steps"]:
            return "answer"
        
        return "continue"  # Continue executing plan


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
        check_cancellation(shared)
        # Get config from shared store
        config = shared.get("config")
        max_results = config.agent.max_query_results if config else 100
        
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "max_results": max_results,
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what simple action to take."""
        prompt = SIMPLE_EXECUTOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        response = call_llm(prompt)
        return {"prompt": prompt, "response": response, "max_results": prep_res["max_results"]}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Execute the simple query."""
        log_thinking(shared, "⚡ Simple Execution", exec_res["response"], level="high")
        
        try:
            query_spec = extract_yaml(exec_res["response"])
            
            client = get_osdk_client()
            # Use config limit or fallback to spec limit
            limit = min(
                query_spec.get("limit", 100),
                exec_res.get("max_results", 100)
            )
            df = client.query_objects(
                object_type=query_spec.get("object_type", ""),
                filters=query_spec.get("filters", {}),
                limit=limit,
            )
            
            shared["fetched_data"] = df
            log_thinking(
                shared,
                "📊 Data Retrieved",
                f"Fetched {len(df)} rows from {query_spec.get('object_type')}",
                level="medium"
            )
            
        except Exception as e:
            log_thinking(shared, "⚠️ Execution Error", str(e), level="low")
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
        check_cancellation(shared)
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
        log_thinking(shared, "💬 Final Answer", exec_res["answer"], level="low")
        return "done"
