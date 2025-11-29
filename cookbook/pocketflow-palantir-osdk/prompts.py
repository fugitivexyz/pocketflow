"""
Prompts for the Palantir OSDK chatbot nodes.
"""

SIMPLE_EXECUTOR_PROMPT = """You are executing a simple data query. Determine the single action needed.

## USER QUERY
{query}

## AVAILABLE OBJECT TYPES
{object_types}

## TASK
Determine the single query needed to answer this. Return in YAML:

```yaml
object_type: <type to query>
filters: {{}}  # any filters needed
limit: 100
description: <what this query does>
```
"""

MULTI_AGENT_ANSWER_PROMPT = """Generate a comprehensive answer based on the multi-agent execution.

## USER QUERY
{query}

## QUERY COMPLEXITY
{query_complexity}

## EXECUTION SUMMARY
{plan_str}

## DATA RETRIEVED
{data_str}

## TASK
Provide a clear, helpful answer that:
1. Directly addresses the user's question
2. Summarizes the key findings
3. Notes any limitations or caveats
4. Is conversational and easy to understand
"""

COORDINATOR_PROMPT = """You are a coordinator that decides how to handle user queries about data.

## USER QUERY
{query}

## AVAILABLE OBJECT TYPES
{object_types}

## DECISION
Analyze the query complexity:

**Simple queries** (handle directly):
- Single object type lookup
- Basic filtering
- Simple counts or listings

**Complex queries** (need planning):
- Multi-step analysis
- Comparisons across multiple dimensions
- Queries requiring data from multiple object types
- Statistical analysis or aggregations
- Visualization requests

Return your decision in YAML:
```yaml
thinking: |
    <analyze the query complexity>
complexity: simple | complex
reason: <why this classification>
```
"""

PLANNER_PROMPT = """You are a query planner. Break down this complex query into executable steps.

## USER QUERY
{query}

## AVAILABLE OBJECT TYPES
{object_types}

## SCHEMAS
{schema_str}

## PLANNING
Create a step-by-step plan. Each step should be one of:
- **fetch**: Query an object type with filters
- **analyze**: Analyze fetched data
- **merge**: Combine data from multiple fetches
- **visualize**: Create a chart (only specify chart_type and title - columns will be auto-inferred from data)
- **answer**: Generate final response

VISUALIZATION NOTES:
- Only specify chart_type and title in chart_spec
- The x, y, and color columns will be automatically inferred based on actual data columns
- Chart types: box (distributions), bar/grouped_bar (comparisons), line (trends), scatter (relationships)

Return the plan in YAML:
```yaml
thinking: |
    <your analysis of what needs to be done>
plan:
  - step: 1
    action: fetch
    object_type: <type>
    filters: {{}}
    description: <what this step does>
  - step: 2
    action: visualize
    chart_spec:
        chart_type: box
        title: <descriptive chart title>
    description: <what this visualization shows>
total_steps: <number>
```
"""
