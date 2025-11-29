"""
PocketFlow nodes for the Palantir OSDK chatbot.

Multi-agent nodes implementing Coordinator-Planner-Executor pattern:
- CoordinatorNode: Routes queries to simple or complex handlers
- PlannerNode: Creates execution plans for complex queries
- ExecutorNode: Executes individual plan steps
- SimpleExecutorNode: Handles simple single-step queries
- AnswerNode: Generates final answers
"""

import yaml
import pandas as pd
import threading
from typing import Dict, Any, Optional

from pocketflow import Node
from utils.call_llm import call_llm, extract_yaml
from utils.osdk_client import get_osdk_client
from utils.visualization import generate_chart, validate_chart_spec, infer_chart_spec
from utils.streaming import log_thinking_streaming
from prompts import COORDINATOR_PROMPT, PLANNER_PROMPT, SIMPLE_EXECUTOR_PROMPT, MULTI_AGENT_ANSWER_PROMPT


# =============================================================================
# Helpers
# =============================================================================

class CancellationError(Exception):
    """Raised when execution is cancelled by user."""
    pass


def check_cancellation(shared: Dict[str, Any]):
    """Check if cancellation has been requested."""
    stop_event = shared.get("stop_event")
    if stop_event and stop_event.is_set():
        raise CancellationError("Execution cancelled by user")


def log_thinking(shared: Dict[str, Any], step_type: str, content: str, level: str = "medium"):
    """
    Log a thinking step for UI transparency.
    
    Args:
        shared: Shared state dictionary
        step_type: Display label (e.g., "🎯 Coordinator")
        content: Content to display
        level: Verbosity level ("low", "medium", "high")
    """
    config = shared.get("config")
    if config:
        verbosity = getattr(config, 'verbosity_level', 'medium')
        levels = {"low": 1, "medium": 2, "high": 3}
        if levels.get(verbosity, 2) < levels.get(level, 2):
            return
    
    log_thinking_streaming(shared, step_type, content)


def get_config_value(shared: Dict[str, Any], attr: str, default: Any) -> Any:
    """Safely get config value with default."""
    config = shared.get("config")
    return getattr(config, attr, default) if config else default


# =============================================================================
# Coordinator Node
# =============================================================================

class CoordinatorNode(Node):
    """Routes queries to simple or complex handlers based on complexity."""
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        prompt = COORDINATOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        response = call_llm(prompt)
        return {"prompt": prompt, "response": response}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "🎯 Coordinator", exec_res["response"], level="high")
        
        try:
            decision = extract_yaml(exec_res["response"])
            complexity = decision.get("complexity", "simple")
        except Exception as e:
            log_thinking(shared, "⚠️ Parse Warning", f"Defaulting to simple: {e}", level="low")
            complexity = "simple"
        
        shared["query_complexity"] = complexity
        return "plan" if complexity == "complex" else "execute_simple"


# =============================================================================
# Planner Node
# =============================================================================

class PlannerNode(Node):
    """Creates execution plans for complex queries."""
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "schemas": shared.get("schemas", {}),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        schema_str = yaml.dump(prep_res["schemas"], default_flow_style=False) if prep_res["schemas"] else "No schemas"
        prompt = PLANNER_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"],
            schema_str=schema_str
        )
        return {"response": call_llm(prompt)}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "📝 Plan Created", exec_res["response"], level="medium")
        
        try:
            plan = extract_yaml(exec_res["response"])
            steps = plan.get("plan", [])
            max_steps = get_config_value(shared, "max_plan_steps", 5)
            if len(steps) > max_steps:
                log_thinking(shared, "⚠️ Plan Truncated", f"Limited to {max_steps} steps", level="low")
                steps = steps[:max_steps]
            shared["execution_plan"] = steps
            shared["current_step"] = 0
        except Exception as e:
            log_thinking(shared, "⚠️ Planning Error", str(e), level="low")
            shared["execution_plan"] = []
        
        return "execute_plan"


# =============================================================================
# Executor Node
# =============================================================================

class ExecutorNode(Node):
    """Executes individual steps from the plan."""
    
    def prep(self, shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        check_cancellation(shared)
        plan = shared.get("execution_plan", [])
        current_step = shared.get("current_step", 0)
        max_steps = get_config_value(shared, "max_plan_steps", 10)
        
        if current_step >= len(plan) or current_step >= max_steps:
            return None
        
        return {
            "step": plan[current_step],
            "step_num": current_step,
            "total_steps": len(plan),
            "accumulated_data": shared.get("accumulated_data", {}),
            "fetched_data": shared.get("fetched_data"),
            "user_query": shared.get("current_query", ""),
            "config": shared.get("config"),
        }

    def exec(self, prep_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if prep_res is None:
            return {"action": "complete"}

        step = prep_res["step"]
        action = step.get("action", "")
        client = get_osdk_client()
        result = {"action": action, "step": step}

        if action == "fetch":
            # Check if pagination is requested
            use_pagination = step.get("paginate", False)
            
            if use_pagination:
                # Use paginated query
                page_result = client.query_objects_paginated(
                    object_type=step.get("object_type", ""),
                    filters=step.get("filters", {}),
                    limit=step.get("limit", 100),
                    offset=step.get("offset", 0),
                    order_by=step.get("order_by"),
                    order_direction=step.get("order_direction", "asc"),
                )
                result["data"] = page_result["data"]
                result["pagination"] = {
                    "total_count": page_result["total_count"],
                    "has_more": page_result["has_more"],
                    "offset": page_result["offset"],
                    "limit": page_result["limit"],
                }
                
                # Auto-paginate if enabled and there's more data
                config = prep_res.get("config")
                if config and getattr(config, "auto_paginate", False) and page_result["has_more"]:
                    all_data = [page_result["data"]]
                    current_offset = page_result["offset"] + page_result["limit"]
                    max_pages = getattr(config, "max_auto_paginate_pages", 10)
                    pages_fetched = 1
                    
                    while page_result["has_more"] and pages_fetched < max_pages:
                        page_result = client.query_objects_paginated(
                            object_type=step.get("object_type", ""),
                            filters=step.get("filters", {}),
                            limit=step.get("limit", 100),
                            offset=current_offset,
                            order_by=step.get("order_by"),
                            order_direction=step.get("order_direction", "asc"),
                        )
                        all_data.append(page_result["data"])
                        current_offset += page_result["limit"]
                        pages_fetched += 1
                    
                    result["data"] = pd.concat(all_data, ignore_index=True)
                    result["pagination"]["pages_fetched"] = pages_fetched
                    result["pagination"]["auto_paginated"] = True
            else:
                # Use simple query (backward compatible)
                df = client.query_objects(
                    object_type=step.get("object_type", ""),
                    filters=step.get("filters", {}),
                    limit=step.get("limit", 100),
                )
                result["data"] = df
            
            result["success"] = True

        elif action == "aggregate":
            # Handle aggregation action
            df = client.aggregate_objects(
                object_type=step.get("object_type", ""),
                group_by=step.get("group_by", []),
                aggregations=step.get("aggregations", {}),
                filters=step.get("filters", {}),
            )
            result["data"] = df
            result["success"] = True

        elif action == "discover_links":
            # Handle link discovery action
            links = client.list_link_types(step.get("object_type", ""))
            result["links"] = links
            result["success"] = True

        elif action == "merge":
            # Handle merge/join action for combining data from multiple fetches
            acc_data = prep_res.get("accumulated_data", {})
            left_key = step.get("left", "")  # e.g., "step_0"
            right_key = step.get("right", "")  # e.g., "step_1"
            on_column = step.get("on", "")  # column to join on
            how = step.get("how", "inner")  # join type: inner, left, right, outer
            
            # Support referencing by step number or key
            left_df = None
            right_df = None
            
            for key, df in acc_data.items():
                if key == left_key or key == f"step_{left_key}" or f"step_{key}" == left_key:
                    left_df = df
                if key == right_key or key == f"step_{right_key}" or f"step_{key}" == right_key:
                    right_df = df
            
            # Also try numeric step references
            if left_df is None and str(left_key).isdigit():
                left_df = acc_data.get(f"step_{left_key}")
            if right_df is None and str(right_key).isdigit():
                right_df = acc_data.get(f"step_{right_key}")
            
            if left_df is not None and right_df is not None:
                try:
                    # Handle case where on_column might have different names
                    left_on = step.get("left_on", on_column)
                    right_on = step.get("right_on", on_column)
                    
                    if left_on and right_on:
                        merged = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
                    elif on_column:
                        merged = pd.merge(left_df, right_df, on=on_column, how=how)
                    else:
                        # Try to find common columns
                        common_cols = list(set(left_df.columns) & set(right_df.columns))
                        if common_cols:
                            merged = pd.merge(left_df, right_df, on=common_cols[0], how=how)
                        else:
                            result["error"] = "No common column found for merge"
                            result["success"] = False
                            return result
                    
                    result["data"] = merged
                    result["success"] = True
                except Exception as e:
                    result["error"] = f"Merge failed: {str(e)}"
                    result["success"] = False
            else:
                missing = []
                if left_df is None:
                    missing.append(f"left ({left_key})")
                if right_df is None:
                    missing.append(f"right ({right_key})")
                result["error"] = f"Missing data for merge: {', '.join(missing)}. Available: {list(acc_data.keys())}"
                result["success"] = False

        elif action == "visualize":
            chart_spec = step.get("chart_spec", {})
            df = prep_res.get("fetched_data")
            
            if df is None:
                acc_data = prep_res.get("accumulated_data", {})
                if acc_data:
                    df = list(acc_data.values())[-1]

            if df is not None and not df.empty:
                try:
                    if not chart_spec.get("x") or not chart_spec.get("y"):
                        validated_spec = infer_chart_spec(df, prep_res.get("user_query", ""), chart_spec)
                    else:
                        validated_spec = validate_chart_spec(chart_spec, df)
                    
                    result["figure"] = generate_chart(df, validated_spec)
                    result["chart_spec"] = validated_spec
                    result["success"] = True
                except Exception as e:
                    result["error"] = str(e)
                    result["success"] = False
            else:
                result["error"] = "No data available"
                result["success"] = False

        elif action in ("analyze", "answer"):
            result["success"] = True

        return result
    
    def post(self, shared: Dict[str, Any], prep_res: Optional[Dict[str, Any]], exec_res: Dict[str, Any]) -> str:
        if exec_res["action"] == "complete":
            return "answer"
        
        step = exec_res.get("step", {})
        log_thinking(
            shared,
            f"⚡ Step {prep_res['step_num'] + 1}/{prep_res['total_steps']}: {exec_res['action']}",
            step.get("description", "N/A"),
            level="medium"
        )
        
        if exec_res["action"] == "fetch" and "data" in exec_res:
            if "accumulated_data" not in shared:
                shared["accumulated_data"] = {}
            shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
            shared["fetched_data"] = exec_res["data"]
            # Log pagination info if available
            if exec_res.get("pagination"):
                pag = exec_res["pagination"]
                pag_info = f"Retrieved {len(exec_res['data'])} of {pag['total_count']} total records"
                if pag.get("auto_paginated"):
                    pag_info += f" (auto-paginated {pag['pages_fetched']} pages)"
                log_thinking(shared, "📄 Pagination", pag_info, level="medium")

        if exec_res["action"] == "aggregate" and "data" in exec_res:
            if "accumulated_data" not in shared:
                shared["accumulated_data"] = {}
            shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
            shared["fetched_data"] = exec_res["data"]
            log_thinking(shared, "📊 Aggregation", f"Computed {len(exec_res['data'])} grouped results", level="medium")

        if exec_res["action"] == "discover_links" and "links" in exec_res:
            links = exec_res["links"]
            shared["discovered_links"] = links
            link_names = [l["link_name"] for l in links]
            log_thinking(shared, "🔗 Links Discovered", f"Found links to: {', '.join(link_names) if link_names else 'none'}", level="medium")

        if exec_res["action"] == "merge":
            if exec_res.get("success") and "data" in exec_res:
                if "accumulated_data" not in shared:
                    shared["accumulated_data"] = {}
                shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
                shared["fetched_data"] = exec_res["data"]
                log_thinking(shared, "🔀 Data Merged", f"Combined into {len(exec_res['data'])} rows", level="medium")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Merge Error", exec_res["error"], level="low")

        if exec_res["action"] == "visualize":
            if exec_res.get("figure"):
                shared["figure"] = exec_res["figure"]
                log_thinking(shared, "📊 Chart Created", str(exec_res.get("chart_spec", {})), level="medium")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Chart Error", exec_res["error"], level="low")

        shared["current_step"] = prep_res["step_num"] + 1
        return "answer" if shared["current_step"] >= prep_res["total_steps"] else "continue"


# =============================================================================
# Simple Executor Node
# =============================================================================

class SimpleExecutorNode(Node):
    """Handles simple single-step queries."""
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "object_types": shared.get("object_types", []),
            "max_results": get_config_value(shared, "max_query_results", 100),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        prompt = SIMPLE_EXECUTOR_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"]
        )
        return {"response": call_llm(prompt), "max_results": prep_res["max_results"]}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "⚡ Simple Execution", exec_res["response"], level="high")
        
        try:
            spec = extract_yaml(exec_res["response"])
            client = get_osdk_client()
            df = client.query_objects(
                object_type=spec.get("object_type", ""),
                filters=spec.get("filters", {}),
                limit=min(spec.get("limit", 100), exec_res["max_results"]),
            )
            shared["fetched_data"] = df
            log_thinking(shared, "📊 Data Retrieved", f"{len(df)} rows from {spec.get('object_type')}", level="medium")
        except Exception as e:
            log_thinking(shared, "⚠️ Error", str(e), level="low")
            shared["error"] = str(e)
        
        return "answer"


# =============================================================================
# Answer Node
# =============================================================================

class AnswerNode(Node):
    """Generates final answers from execution results."""
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        check_cancellation(shared)
        return {
            "query": shared.get("current_query", ""),
            "fetched_data": shared.get("fetched_data"),
            "accumulated_data": shared.get("accumulated_data", {}),
            "execution_plan": shared.get("execution_plan", []),
            "query_complexity": shared.get("query_complexity", "simple"),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        data_context = []
        
        if prep_res["fetched_data"] is not None:
            df = prep_res["fetched_data"]
            data_context.append(f"Main data: {len(df)} rows, columns: {list(df.columns)}")
            data_context.append(f"Sample:\n{df.head(5).to_string()}")
        
        for key, df in prep_res["accumulated_data"].items():
            if df is not None and not df.empty:
                data_context.append(f"\n{key}: {len(df)} rows")
        
        data_str = "\n".join(data_context) if data_context else "No data retrieved."
        plan_str = f"Executed {len(prep_res['execution_plan'])} steps." if prep_res["execution_plan"] else "Simple query."
        
        prompt = MULTI_AGENT_ANSWER_PROMPT.format(
            query=prep_res["query"],
            query_complexity=prep_res["query_complexity"],
            plan_str=plan_str,
            data_str=data_str
        )
        return {"answer": call_llm(prompt)}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        shared["final_answer"] = exec_res["answer"]
        log_thinking(shared, "💬 Final Answer", exec_res["answer"], level="low")
        return "done"
