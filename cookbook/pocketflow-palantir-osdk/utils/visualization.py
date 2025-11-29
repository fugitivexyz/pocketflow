"""
Visualization utilities for generating Plotly charts.

The LLM generates chart specifications in a structured format,
and this module converts them to Plotly figures.
"""

import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, Optional, Union


# Valid chart types and aggregations (whitelists)
VALID_CHART_TYPES = {"bar", "grouped_bar", "line", "scatter", "heatmap", "box"}
VALID_AGGREGATIONS = {"mean", "sum", "count", "median"}


def infer_chart_spec(
    df: pd.DataFrame,
    user_query: str,
    partial_spec: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Use LLM to infer the best chart specification based on actual data.

    This function dynamically determines the best columns for visualization
    by analyzing the DataFrame structure and the user's query intent.

    Args:
        df: The DataFrame to visualize
        user_query: The original user question (for context)
        partial_spec: Optional partial chart_spec from planner (may have chart_type, title)

    Returns:
        Complete, validated chart_spec ready for generate_chart()
    """
    from utils.call_llm import call_llm, extract_yaml

    # Build data context for LLM
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().head(3).tolist()
        columns_info.append(f"  - {col}: {dtype}, {n_unique} unique values, examples: {sample_vals}")

    columns_str = "\n".join(columns_info)

    # Include partial spec if provided
    partial_spec_str = "None provided"
    if partial_spec:
        partial_spec_str = yaml.dump(partial_spec, default_flow_style=False)

    prompt = f"""You are a data visualization expert. Given a DataFrame and user query, determine the best chart configuration.

## USER QUERY
{user_query}

## DATAFRAME INFO
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns:
{columns_str}

## PARTIAL CHART SPEC (if any)
{partial_spec_str}

## TASK
Select the most appropriate columns for visualization. Consider:
1. For X-axis: Pick a categorical column (like sample_id, experiment_name) or a time/sequence column
2. For Y-axis: Pick a numeric column that answers the user's question (e.g., turbidity, value, concentration)
3. For Color grouping: Pick a categorical column to compare groups (e.g., surfactant, treatment)
4. For aggregation: Use "mean" when multiple measurements exist per group

## CHART TYPES
- bar: Compare categories
- grouped_bar: Compare categories with sub-groups
- line: Show trends over time/sequence
- box: Show distributions at each category (BEST for multiple samples per group)
- scatter: Show relationships between two numeric values

Return ONLY valid YAML with column names that EXACTLY match the DataFrame columns above:
```yaml
chart_type: <bar|grouped_bar|line|box|scatter>
x: <exact column name for x-axis>
y: <exact column name for y-axis>
color: <exact column name for grouping, or null if not needed>
aggregation: <mean|sum|median|null>
title: <descriptive title>
```
"""

    response = call_llm(prompt)
    inferred_spec = extract_yaml(response)

    # Validate the inferred spec against actual columns
    validated_spec = validate_chart_spec(inferred_spec, df)

    return validated_spec


def validate_chart_spec(chart_spec: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate and sanitize chart_spec from LLM output.

    This function ensures that chart specifications are valid and safe to use,
    preventing errors from invalid column names, None values, or unsupported types.

    Args:
        chart_spec: Raw chart specification from LLM
        df: DataFrame to validate column names against

    Returns:
        Validated and sanitized chart specification

    Raises:
        ValueError: If required columns (x, y) are missing or invalid
    """
    validated = {}
    available_columns = list(df.columns)

    # Validate chart type (default to "bar" if invalid)
    chart_type = chart_spec.get("chart_type", "bar")
    if chart_type not in VALID_CHART_TYPES:
        chart_type = "bar"
    validated["chart_type"] = chart_type

    # Validate required columns (x, y)
    for col_name in ["x", "y"]:
        val = chart_spec.get(col_name)
        if val is None:
            raise ValueError(f"Required parameter '{col_name}' is missing")
        if val not in available_columns:
            raise ValueError(f"Column '{val}' not found. Available columns: {available_columns}")
        validated[col_name] = val

    # Validate optional column parameters (skip if None or invalid)
    for col_name in ["color", "facet_col", "facet_row"]:
        val = chart_spec.get(col_name)
        if val is not None and val in available_columns:
            validated[col_name] = val
        # If None or invalid column, simply don't include in validated spec

    # Validate aggregation
    agg = chart_spec.get("aggregation")
    if agg is not None and agg in VALID_AGGREGATIONS:
        validated["aggregation"] = agg

    # Sanitize string parameters (title, labels) - limit length to prevent issues
    for param in ["title", "x_label", "y_label"]:
        val = chart_spec.get(param)
        if val is not None and isinstance(val, str):
            validated[param] = val[:200]  # Limit length for safety
        elif param == "title":
            validated["title"] = ""  # Default empty title

    return validated


def generate_chart(
    data: pd.DataFrame,
    chart_spec: Dict[str, Any],
    template: str = "plotly_white",
) -> go.Figure:
    """
    Generate a Plotly chart from data and a specification.
    
    Args:
        data: DataFrame with the data to visualize
        chart_spec: Dict with chart configuration:
            - chart_type: "bar", "grouped_bar", "line", "scatter", "heatmap", "box"
            - x: Column name for x-axis
            - y: Column name for y-axis
            - color: (Optional) Column name for color grouping
            - title: (Optional) Chart title
            - x_label: (Optional) X-axis label
            - y_label: (Optional) Y-axis label
            - facet_col: (Optional) Column for faceting into subplots
            - facet_row: (Optional) Row faceting
            - aggregation: (Optional) "mean", "sum", "count", "median"
        template: Plotly template name for chart styling (default: "plotly_white")
            
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
        if color:
            group_cols.append(color)
        if facet_col:
            group_cols.append(facet_col)
        if facet_row:
            group_cols.append(facet_row)
        
        if aggregation == "mean":
            data = data.groupby(group_cols, as_index=False)[y].mean()
        elif aggregation == "sum":
            data = data.groupby(group_cols, as_index=False)[y].sum()
        elif aggregation == "count":
            data = data.groupby(group_cols, as_index=False)[y].count()
        elif aggregation == "median":
            data = data.groupby(group_cols, as_index=False)[y].median()
    
    # Generate chart based on type
    fig = None
    
    if chart_type == "bar":
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            barmode="group" if color else "relative",
        )
    
    elif chart_type == "grouped_bar":
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            barmode="group",
        )
    
    elif chart_type == "line":
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
            markers=True,
        )
    
    elif chart_type == "scatter":
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
        )
    
    elif chart_type == "heatmap":
        # For heatmap, we need to pivot the data
        if color:
            pivot_data = data.pivot_table(
                index=y, columns=x, values=color, aggfunc="mean"
            )
        else:
            pivot_data = data.pivot_table(
                index=y, columns=x, values=data.columns[-1], aggfunc="mean"
            )
        
        fig = px.imshow(
            pivot_data,
            title=title,
            aspect="auto",
            color_continuous_scale="RdBu_r",
        )
    
    elif chart_type == "box":
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            facet_col=facet_col,
            facet_row=facet_row,
        )
    
    else:
        # Default to bar chart
        fig = px.bar(data, x=x, y=y, color=color, title=title)
    
    # Update layout
    if fig:
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=color if color else None,
            template=template,
        )
    
    return fig


def create_comparison_chart(
    data: pd.DataFrame,
    group_by: str,
    compare_by: str,
    value_col: str,
    title: str = "Comparison Chart",
    aggregation: str = "mean",
) -> go.Figure:
    """
    Create a comparison chart for analyzing groups.
    
    Useful for questions like:
    "Compare all surfactants across different concentrations"
    
    Args:
        data: DataFrame with the data
        group_by: Primary grouping column (e.g., "surfactant")
        compare_by: Secondary comparison column (e.g., "surfactant_concentration")
        value_col: Column with values to compare
        title: Chart title
        aggregation: How to aggregate values ("mean", "sum", "median")
        
    Returns:
        Plotly Figure
    """
    # Aggregate data
    if aggregation == "mean":
        agg_data = data.groupby([group_by, compare_by], as_index=False)[value_col].mean()
    elif aggregation == "sum":
        agg_data = data.groupby([group_by, compare_by], as_index=False)[value_col].sum()
    elif aggregation == "median":
        agg_data = data.groupby([group_by, compare_by], as_index=False)[value_col].median()
    else:
        agg_data = data.groupby([group_by, compare_by], as_index=False)[value_col].mean()
    
    fig = px.bar(
        agg_data,
        x=group_by,
        y=value_col,
        color=compare_by,
        title=title,
        barmode="group",
    )
    
    fig.update_layout(template="plotly_white")
    return fig


def create_time_series_chart(
    data: pd.DataFrame,
    time_col: str,
    value_col: str,
    group_col: Optional[str] = None,
    title: str = "Time Series",
) -> go.Figure:
    """
    Create a time series chart for tracking measurements over time.
    
    Args:
        data: DataFrame with the data
        time_col: Column with time points (e.g., "time_point")
        value_col: Column with values to plot
        group_col: Optional column for multiple series
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    # Sort by time for proper line chart
    data = data.sort_values(time_col)
    
    fig = px.line(
        data,
        x=time_col,
        y=value_col,
        color=group_col,
        title=title,
        markers=True,
    )
    
    fig.update_layout(template="plotly_white")
    return fig


# Test the visualization utilities
if __name__ == "__main__":
    # Create sample data
    import numpy as np
    
    np.random.seed(42)
    test_data = pd.DataFrame({
        "surfactant": ["PS80"] * 6 + ["PS20"] * 6 + ["Poloxamer"] * 6,
        "concentration": [0.01, 0.02, 0.05] * 6,
        "time_point": ["D0", "D7"] * 9,
        "turbidity": np.random.uniform(0.1, 0.5, 18),
    })
    
    print("## Test data:")
    print(test_data)
    
    # Test bar chart
    spec = {
        "chart_type": "grouped_bar",
        "x": "surfactant",
        "y": "turbidity",
        "color": "concentration",
        "title": "Turbidity by Surfactant and Concentration",
        "aggregation": "mean",
    }
    
    fig = generate_chart(test_data, spec)
    print("\n## Generated grouped bar chart")
    fig.show()
    
    # Test comparison chart
    fig2 = create_comparison_chart(
        test_data,
        group_by="surfactant",
        compare_by="time_point",
        value_col="turbidity",
        title="Turbidity Comparison",
    )
    print("\n## Generated comparison chart")
    fig2.show()
