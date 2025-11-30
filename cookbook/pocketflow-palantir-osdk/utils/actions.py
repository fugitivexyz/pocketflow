"""
Action execution utilities for PocketFlow nodes.

Provides:
- execute_fetch: Query objects with optional pagination
- execute_aggregate: Compute aggregations on objects
- execute_discover_links: Find available link types
- execute_merge: Join data from multiple fetches
- execute_visualize: Generate charts from data

Each function takes step parameters and returns a result dict.
Per PocketFlow philosophy, no exception handling - let Node retry handle failures.
"""

import pandas as pd
from typing import Dict, Any, Optional

try:
    from utils.osdk_client import get_osdk_client
    from utils.visualization import generate_chart, validate_chart_spec, infer_chart_spec
except ImportError:
    from osdk_client import get_osdk_client
    from visualization import generate_chart, validate_chart_spec, infer_chart_spec


def execute_fetch(step: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Execute a fetch action to query objects.
    
    Args:
        step: Step definition with object_type, filters, limit, paginate, etc.
        config: Optional config for auto-pagination settings
        
    Returns:
        Dict with data, pagination info, and success status
    """
    client = get_osdk_client()
    result = {"action": "fetch", "step": step}
    use_pagination = step.get("paginate", False)
    
    if use_pagination:
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
    return result


def execute_aggregate(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an aggregate action to compute statistics.
    
    Args:
        step: Step definition with object_type, group_by, aggregations, filters
        
    Returns:
        Dict with aggregated data and success status
    """
    client = get_osdk_client()
    df = client.aggregate_objects(
        object_type=step.get("object_type", ""),
        group_by=step.get("group_by", []),
        aggregations=step.get("aggregations", {}),
        filters=step.get("filters", {}),
    )
    return {
        "action": "aggregate",
        "step": step,
        "data": df,
        "success": True,
    }


def execute_discover_links(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a discover_links action to find available relationships.
    
    Args:
        step: Step definition with object_type
        
    Returns:
        Dict with links list and success status
    """
    client = get_osdk_client()
    links = client.list_link_types(step.get("object_type", ""))
    return {
        "action": "discover_links",
        "step": step,
        "links": links,
        "success": True,
    }


def execute_merge(step: Dict[str, Any], accumulated_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Execute a merge action to join data from multiple fetches.
    
    Args:
        step: Step definition with left, right, on, how, left_on, right_on
        accumulated_data: Dict of step_id -> DataFrame from previous steps
        
    Returns:
        Dict with merged data and success status
    """
    result = {"action": "merge", "step": step}
    
    left_key = step.get("left", "")
    right_key = step.get("right", "")
    on_column = step.get("on", "")
    how = step.get("how", "inner")
    
    # Resolve DataFrames from accumulated data
    left_df = _resolve_dataframe(left_key, accumulated_data)
    right_df = _resolve_dataframe(right_key, accumulated_data)
    
    if left_df is None or right_df is None:
        missing = []
        if left_df is None:
            missing.append(f"left ({left_key})")
        if right_df is None:
            missing.append(f"right ({right_key})")
        result["error"] = f"Missing data for merge: {', '.join(missing)}. Available: {list(accumulated_data.keys())}"
        result["success"] = False
        return result
    
    # Perform merge
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
    return result


def _resolve_dataframe(key: Any, accumulated_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Resolve a DataFrame from accumulated data by key.
    
    Supports referencing by:
    - Direct key: "step_0"
    - Step number: 0
    - String number: "0"
    """
    # Direct match
    if key in accumulated_data:
        return accumulated_data[key]
    
    # Try step_X format
    step_key = f"step_{key}"
    if step_key in accumulated_data:
        return accumulated_data[step_key]
    
    # Reverse lookup
    for data_key, df in accumulated_data.items():
        if data_key == key or data_key == step_key or f"step_{data_key}" == key:
            return df
    
    # Numeric reference
    if str(key).isdigit():
        return accumulated_data.get(f"step_{key}")
    
    return None


def execute_visualize(
    step: Dict[str, Any],
    fetched_data: Optional[pd.DataFrame],
    accumulated_data: Dict[str, pd.DataFrame],
    user_query: str = "",
) -> Dict[str, Any]:
    """
    Execute a visualize action to generate a chart.
    
    Args:
        step: Step definition with chart_spec
        fetched_data: Most recent fetched DataFrame
        accumulated_data: Dict of step_id -> DataFrame from previous steps
        user_query: Original user query for chart inference
        
    Returns:
        Dict with figure, chart_spec, and success status
    """
    result = {"action": "visualize", "step": step}
    chart_spec = step.get("chart_spec", {})
    
    # Get data source
    df = fetched_data
    if df is None and accumulated_data:
        df = list(accumulated_data.values())[-1]
    
    if df is None or df.empty:
        result["error"] = "No data available for visualization"
        result["success"] = False
        return result
    
    # Infer or validate chart spec
    if not chart_spec.get("x") or not chart_spec.get("y"):
        validated_spec = infer_chart_spec(df, user_query, chart_spec)
    else:
        validated_spec = validate_chart_spec(chart_spec, df)
    
    result["figure"] = generate_chart(df, validated_spec)
    result["chart_spec"] = validated_spec
    result["success"] = True
    return result


def dispatch_action(
    action: str,
    step: Dict[str, Any],
    prep_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dispatch to the appropriate action executor.
    
    Args:
        action: Action type (fetch, aggregate, discover_links, merge, visualize, analyze, answer)
        step: Step definition from the plan
        prep_res: Prepared resources from node prep
        
    Returns:
        Result dict from the action executor
    """
    if action == "fetch":
        return execute_fetch(step, prep_res.get("config"))
    
    elif action == "aggregate":
        return execute_aggregate(step)
    
    elif action == "discover_links":
        return execute_discover_links(step)
    
    elif action == "merge":
        return execute_merge(step, prep_res.get("accumulated_data", {}))
    
    elif action == "visualize":
        return execute_visualize(
            step,
            prep_res.get("fetched_data"),
            prep_res.get("accumulated_data", {}),
            prep_res.get("user_query", ""),
        )
    
    elif action in ("analyze", "answer"):
        return {"action": action, "step": step, "success": True}
    
    else:
        return {"action": action, "step": step, "success": False, "error": f"Unknown action: {action}"}


if __name__ == "__main__":
    import os
    
    # Set up mock OSDK
    os.environ["USE_MOCK_OSDK"] = "true"
    
    print("Testing action utilities...")
    
    # Test execute_fetch
    print("\n1. Testing execute_fetch...")
    result = execute_fetch({"object_type": "Experiment", "filters": {}, "limit": 10})
    assert result["success"], "Fetch should succeed"
    assert "data" in result, "Fetch should return data"
    print(f"   ✅ Fetched {len(result['data'])} rows")
    
    # Test execute_aggregate
    print("\n2. Testing execute_aggregate...")
    result = execute_aggregate({
        "object_type": "Result",
        "group_by": ["measurement_type"],
        "aggregations": {"value": "mean"},
        "filters": {},
    })
    assert result["success"], "Aggregate should succeed"
    print(f"   ✅ Aggregated {len(result['data'])} groups")
    
    # Test execute_discover_links
    print("\n3. Testing execute_discover_links...")
    result = execute_discover_links({"object_type": "Sample"})
    assert result["success"], "Discover links should succeed"
    print(f"   ✅ Found {len(result['links'])} link types")
    
    # Test execute_merge
    print("\n4. Testing execute_merge...")
    accumulated = {
        "step_0": pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        "step_1": pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
    }
    result = execute_merge({"left": 0, "right": 1, "on": "id"}, accumulated)
    assert result["success"], "Merge should succeed"
    assert len(result["data"]) == 3, "Merge should have 3 rows"
    print(f"   ✅ Merged to {len(result['data'])} rows with columns: {list(result['data'].columns)}")
    
    # Test dispatch_action
    print("\n5. Testing dispatch_action...")
    result = dispatch_action("fetch", {"object_type": "Experiment", "filters": {}}, {})
    assert result["success"], "Dispatch should work"
    print("   ✅ Dispatch works correctly")
    
    print("\n✅ All action tests passed!")
