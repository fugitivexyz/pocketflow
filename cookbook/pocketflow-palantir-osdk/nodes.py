"""
PocketFlow nodes for the Palantir OSDK chatbot.

This module contains multi-agent nodes: Coordinator, Planner, Executor

All nodes log their prompts and responses to shared["thinking_steps"] for UI transparency.
"""

from pocketflow import Node, Flow
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.call_llm import call_llm, extract_yaml
from utils.osdk_client import get_osdk_client
from utils.visualization import generate_chart
from utils.streaming import log_thinking_streaming
from prompts import COORDINATOR_PROMPT, PLANNER_PROMPT


def log_thinking(shared: Dict[str, Any], step_type: str, content: str):
    """
    Log a thinking step for UI transparency.
    
    This function now supports real-time streaming to the UI
    via the streaming callback mechanism.
    """
    log_thinking_streaming(shared, step_type, content)


# =============================================================================
# Multi-Agent Nodes
# =============================================================================

class CoordinatorNode(Node):
    """
    Coordinator for multi-agent system.
    
    Decides whether to:
    - Handle simple queries directly
    - Dispatch to Planner for complex multi-step queries
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Get the query and context."""
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "conversation_history": shared.get("messages", []),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Decide if query is simple or complex."""
        prompt = COORDINATOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        
        response = call_llm(prompt)
        return {"prompt": prompt, "response": response}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Route to appropriate handler."""
        log_thinking(shared, "🎯 Coordinator", exec_res["response"])
        
        try:
            decision = extract_yaml(exec_res["response"])
            complexity = decision.get("complexity", "simple")
        except:
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
    """
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Get query and available schemas."""
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "schemas": shared.get("schemas", {}),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan."""
        schema_str = yaml.dump(prep_res["schemas"], default_flow_style=False) if prep_res["schemas"] else "No schemas loaded"
        
        prompt = PLANNER_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"],
            schema_str=schema_str
        )
        
        response = call_llm(prompt)
        return {"prompt": prompt, "response": response}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        """Store the plan and proceed to execution."""
        log_thinking(shared, "📝 Plan Created", exec_res["response"])
        
        try:
            plan = extract_yaml(exec_res["response"])
            shared["execution_plan"] = plan.get("plan", [])
            shared["current_step"] = 0
        except Exception as e:
            log_thinking(shared, "⚠️ Planning Error", str(e))
            shared["execution_plan"] = []
        
        return "execute_plan"


class ExecutorNode(Node):
    """
    Executor for multi-agent system.
    
    Executes individual steps from the plan.
    """
    
    def prep(self, shared):
        """Get current step from plan."""
        plan = shared.get("execution_plan", [])
        current_step = shared.get("current_step", 0)

        if current_step >= len(plan):
            return None  # Plan complete

        return {
            "step": plan[current_step],
            "step_num": current_step,
            "total_steps": len(plan),
            "accumulated_data": shared.get("accumulated_data", {}),
            "fetched_data": shared.get("fetched_data"),  # Current fetched data
            "user_query": shared.get("current_query", ""),  # For dynamic chart inference
        }

    def exec(self, prep_res):
        """Execute the current step."""
        if prep_res is None:
            return {"action": "complete", "result": None}

        step = prep_res["step"]
        action = step.get("action", "")
        client = get_osdk_client()

        result = {"action": action, "step": step}

        if action == "fetch":
            df = client.query_objects(
                object_type=step.get("object_type", ""),
                filters=step.get("filters", {}),
                limit=step.get("limit", 100),
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

                    fig = generate_chart(df, validated_spec)
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
            f"Description: {step.get('description', 'N/A')}"
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
                log_thinking(shared, "�� Visualization Created",
                    f"Chart spec: {exec_res.get('chart_spec', {})}")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Visualization Error", exec_res["error"])

        # Advance to next step
        shared["current_step"] = prep_res["step_num"] + 1
        
        # Check if more steps
        if shared["current_step"] >= prep_res["total_steps"]:
            return "answer"
        
        return "continue"  # Continue executing plan
