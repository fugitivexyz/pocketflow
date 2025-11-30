"""
PocketFlow nodes for the Palantir OSDK chatbot.

Multi-agent nodes implementing Coordinator-Planner-Executor pattern:
- CoordinatorNode: Routes queries to simple or complex handlers
- PlannerNode: Creates execution plans for complex queries
- ExecutorNode: Executes individual plan steps
- SimpleExecutorNode: Handles simple single-step queries
- AnswerNode: Generates final answers

Per PocketFlow philosophy:
- Nodes are thin orchestration layers
- exec() is for pure computation (LLM/API calls)
- post() handles shared store writes
- Exception handling uses Node's built-in retry mechanism
"""

import yaml
from typing import Dict, Any, Optional

from pocketflow import Node
from utils.call_llm import call_llm, extract_yaml
from utils.osdk_client import get_osdk_client
from utils.helpers import check_cancellation, log_thinking, get_config_value
from utils.actions import dispatch_action
from prompts import COORDINATOR_PROMPT, PLANNER_PROMPT, SIMPLE_EXECUTOR_PROMPT, MULTI_AGENT_ANSWER_PROMPT


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
        return {"response": response, "decision": extract_yaml(response)}
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """On failure, default to simple execution."""
        return {"response": "", "decision": {"complexity": "simple"}, "fallback_error": str(exc)}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "🎯 Coordinator", exec_res.get("response", ""), level="high")
        
        if exec_res.get("fallback_error"):
            log_thinking(shared, "⚠️ Parse Warning", f"Defaulting to simple: {exec_res['fallback_error']}", level="low")
        
        complexity = exec_res["decision"].get("complexity", "simple")
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
            "max_steps": get_config_value(shared, "max_plan_steps", 5),
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        schema_str = yaml.dump(prep_res["schemas"], default_flow_style=False) if prep_res["schemas"] else "No schemas"
        prompt = PLANNER_PROMPT.format(
            query=prep_res["query"],
            object_types=prep_res["object_types"],
            schema_str=schema_str
        )
        response = call_llm(prompt)
        return {"response": response, "plan": extract_yaml(response)}
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """On failure, return empty plan."""
        return {"response": "", "plan": {"plan": []}, "fallback_error": str(exc)}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "📝 Plan Created", exec_res.get("response", ""), level="medium")
        
        if exec_res.get("fallback_error"):
            log_thinking(shared, "⚠️ Planning Error", exec_res["fallback_error"], level="low")
        
        steps = exec_res["plan"].get("plan", [])
        max_steps = prep_res["max_steps"]
        if len(steps) > max_steps:
            log_thinking(shared, "⚠️ Plan Truncated", f"Limited to {max_steps} steps", level="low")
            steps = steps[:max_steps]
        
        shared["execution_plan"] = steps
        shared["current_step"] = 0
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
        return dispatch_action(action, step, prep_res)
    
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
        
        # Update shared store based on action results
        self._update_shared_from_result(shared, prep_res, exec_res)
        
        shared["current_step"] = prep_res["step_num"] + 1
        return "answer" if shared["current_step"] >= prep_res["total_steps"] else "continue"
    
    def _update_shared_from_result(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> None:
        """Update shared store based on action execution result."""
        action = exec_res["action"]
        
        if action == "fetch" and "data" in exec_res:
            if "accumulated_data" not in shared:
                shared["accumulated_data"] = {}
            shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
            shared["fetched_data"] = exec_res["data"]
            if exec_res.get("pagination"):
                pag = exec_res["pagination"]
                pag_info = f"Retrieved {len(exec_res['data'])} of {pag['total_count']} total records"
                if pag.get("auto_paginated"):
                    pag_info += f" (auto-paginated {pag['pages_fetched']} pages)"
                log_thinking(shared, "📄 Pagination", pag_info, level="medium")

        elif action == "aggregate" and "data" in exec_res:
            if "accumulated_data" not in shared:
                shared["accumulated_data"] = {}
            shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
            shared["fetched_data"] = exec_res["data"]
            log_thinking(shared, "📊 Aggregation", f"Computed {len(exec_res['data'])} grouped results", level="medium")

        elif action == "discover_links" and "links" in exec_res:
            links = exec_res["links"]
            shared["discovered_links"] = links
            link_names = [l["link_name"] for l in links]
            log_thinking(shared, "🔗 Links Discovered", f"Found links to: {', '.join(link_names) if link_names else 'none'}", level="medium")

        elif action == "merge":
            if exec_res.get("success") and "data" in exec_res:
                if "accumulated_data" not in shared:
                    shared["accumulated_data"] = {}
                shared["accumulated_data"][f"step_{prep_res['step_num']}"] = exec_res["data"]
                shared["fetched_data"] = exec_res["data"]
                log_thinking(shared, "🔀 Data Merged", f"Combined into {len(exec_res['data'])} rows", level="medium")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Merge Error", exec_res["error"], level="low")

        elif action == "visualize":
            if exec_res.get("figure"):
                shared["figure"] = exec_res["figure"]
                log_thinking(shared, "📊 Chart Created", str(exec_res.get("chart_spec", {})), level="medium")
            elif exec_res.get("error"):
                log_thinking(shared, "⚠️ Chart Error", exec_res["error"], level="low")


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
        response = call_llm(prompt)
        spec = extract_yaml(response)
        
        client = get_osdk_client()
        df = client.query_objects(
            object_type=spec.get("object_type", ""),
            filters=spec.get("filters", {}),
            limit=min(spec.get("limit", 100), prep_res["max_results"]),
        )
        return {"response": response, "spec": spec, "data": df}
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """On failure, return error info."""
        return {"response": "", "spec": {}, "data": None, "error": str(exc)}
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        log_thinking(shared, "⚡ Simple Execution", exec_res.get("response", ""), level="high")
        
        if exec_res.get("error"):
            log_thinking(shared, "⚠️ Error", exec_res["error"], level="low")
            shared["error"] = exec_res["error"]
        elif exec_res.get("data") is not None:
            shared["fetched_data"] = exec_res["data"]
            log_thinking(shared, "📊 Data Retrieved", f"{len(exec_res['data'])} rows from {exec_res['spec'].get('object_type')}", level="medium")
        
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
