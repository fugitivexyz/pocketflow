"""
Visualization utilities for generating Plotly charts.

Converts structured chart specifications to Plotly figures.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, Optional


VALID_CHART_TYPES = {"bar", "grouped_bar", "line", "scatter", "heatmap", "box"}
VALID_AGGREGATIONS = {"mean", "sum", "count", "median"}


def validate_chart_spec(chart_spec: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate and sanitize chart_spec against DataFrame columns.
    
    Raises ValueError if required columns (x, y) are missing or invalid.
    """
    validated = {}
    available_columns = list(df.columns)

    # Validate chart type
    chart_type = chart_spec.get("chart_type", "bar")
    validated["chart_type"] = chart_type if chart_type in VALID_CHART_TYPES else "bar"

    # Validate required columns
    for col_name in ["x", "y"]:
        val = chart_spec.get(col_name)
        if val is None:
            raise ValueError(f"Required parameter '{col_name}' is missing")
        if val not in available_columns:
            raise ValueError(f"Column '{val}' not found. Available: {available_columns}")
        validated[col_name] = val

    # Validate optional columns
    for col_name in ["color", "facet_col", "facet_row"]:
        val = chart_spec.get(col_name)
        if val is not None and val in available_columns:
            validated[col_name] = val

    # Validate aggregation
    agg = chart_spec.get("aggregation")
    if agg in VALID_AGGREGATIONS:
        validated["aggregation"] = agg

    # Sanitize string parameters
    for param in ["title", "x_label", "y_label"]:
        val = chart_spec.get(param)
        if val and isinstance(val, str):
            validated[param] = val[:200]
    
    if "title" not in validated:
        validated["title"] = ""

    return validated


def generate_chart(
    data: pd.DataFrame,
    chart_spec: Dict[str, Any],
    template: str = "plotly_white",
) -> go.Figure:
    """
    Generate a Plotly chart from data and specification.
    
    Args:
        data: DataFrame with data to visualize
        chart_spec: Dict with chart configuration:
            - chart_type: "bar", "grouped_bar", "line", "scatter", "heatmap", "box"
            - x: Column name for x-axis
            - y: Column name for y-axis
            - color: (Optional) Column for color grouping
            - title: (Optional) Chart title
            - aggregation: (Optional) "mean", "sum", "count", "median"
        template: Plotly template name
            
    Returns:
        Plotly Figure object
    """
    chart_type = chart_spec.get("chart_type", "bar")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    color = chart_spec.get("color")
    title = chart_spec.get("title", "")
    x_label = chart_spec.get("x_label", x)
    y_label = chart_spec.get("y_label", y)
    facet_col = chart_spec.get("facet_col")
    facet_row = chart_spec.get("facet_row")
    aggregation = chart_spec.get("aggregation")
    
    # Apply aggregation if specified
    if aggregation and x:
        group_cols = [x]
        for col in [color, facet_col, facet_row]:
            if col:
                group_cols.append(col)
        
        agg_func = {"mean": "mean", "sum": "sum", "count": "count", "median": "median"}
        data = data.groupby(group_cols, as_index=False)[y].agg(agg_func.get(aggregation, "mean"))
    
    # Generate chart
    fig = None
    common_args = dict(x=x, y=y, color=color, title=title, facet_col=facet_col, facet_row=facet_row)
    
    if chart_type == "bar":
        fig = px.bar(data, **common_args, barmode="group" if color else "relative")
    elif chart_type == "grouped_bar":
        fig = px.bar(data, **common_args, barmode="group")
    elif chart_type == "line":
        fig = px.line(data, **common_args, markers=True)
    elif chart_type == "scatter":
        fig = px.scatter(data, **common_args)
    elif chart_type == "box":
        fig = px.box(data, **common_args)
    elif chart_type == "heatmap":
        pivot_col = color if color else data.columns[-1]
        pivot_data = data.pivot_table(index=y, columns=x, values=pivot_col, aggfunc="mean")
        fig = px.imshow(pivot_data, title=title, aspect="auto", color_continuous_scale="RdBu_r")
    else:
        fig = px.bar(data, x=x, y=y, color=color, title=title)
    
    if fig:
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=color if color else None,
            template=template,
        )
    
    return fig


def infer_chart_spec(df: pd.DataFrame, user_query: str, partial_spec: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Use LLM to infer the best chart specification based on actual data.
    
    Args:
        df: DataFrame to visualize
        user_query: Original user question for context
        partial_spec: Partial chart_spec from planner (may have chart_type, title)
    
    Returns:
        Complete, validated chart_spec
    """
    import yaml
    from utils.call_llm import call_llm, extract_yaml as parse_yaml

    # Build data context
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().head(3).tolist()
        columns_info.append(f"  - {col}: {dtype}, {n_unique} unique, examples: {sample_vals}")

    partial_str = yaml.dump(partial_spec, default_flow_style=False) if partial_spec else "None"

    prompt = f"""You are a data visualization expert. Select appropriate columns for a chart.

## USER QUERY
{user_query}

## DATAFRAME
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns:
{chr(10).join(columns_info)}

## PARTIAL SPEC
{partial_str}

## CHART TYPES
- bar/grouped_bar: Compare categories
- line: Trends over time
- box: Distributions (BEST for multiple samples per group)
- scatter: Two numeric variables

Return ONLY valid YAML with column names that EXACTLY match above:
```yaml
chart_type: <bar|grouped_bar|line|box|scatter>
x: <exact column name>
y: <exact column name>
color: <exact column name or null>
aggregation: <mean|sum|median|null>
title: <title>
```
"""

    response = call_llm(prompt)
    inferred = parse_yaml(response)
    return validate_chart_spec(inferred, df)


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        "surfactant": ["PS80"] * 6 + ["PS20"] * 6,
        "concentration": [0.01, 0.02, 0.05] * 4,
        "turbidity": np.random.uniform(0.1, 0.5, 12),
    })
    
    spec = {
        "chart_type": "grouped_bar",
        "x": "surfactant",
        "y": "turbidity",
        "color": "concentration",
        "title": "Turbidity by Surfactant",
        "aggregation": "mean",
    }
    
    fig = generate_chart(test_data, spec)
    print("Chart generated successfully")
    fig.show()
