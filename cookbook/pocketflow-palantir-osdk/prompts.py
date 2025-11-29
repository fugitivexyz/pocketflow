"""
Prompts for the Palantir OSDK chatbot nodes.
"""

SIMPLE_EXECUTOR_PROMPT = """You are executing a simple data query. Determine the single action needed.

## USER QUERY
{query}

## AVAILABLE OBJECT TYPES
{object_types}

## FILTER SYNTAX
You can use rich filter operators:
- Simple equality: {{"status": "completed"}}
- Comparison: {{"value": {{"$gt": 50}}}}, {{"value": {{"$lt": 100}}}}, $gte, $lte
- Not equal: {{"status": {{"$ne": "cancelled"}}}}
- List membership: {{"surfactant": {{"$in": ["PS80", "PS20"]}}}}
- Not in list: {{"status": {{"$nin": ["cancelled", "failed"]}}}}
- Text contains: {{"name": {{"$contains": "stability"}}}}
- Starts with: {{"sample_id": {{"$startswith": "SAM00"}}}}
- Combine operators: {{"value": {{"$gt": 10, "$lt": 50}}}}

## TASK
Determine the single query needed to answer this. Return in YAML:

```yaml
object_type: <type to query>
filters: {{}}  # use rich filter operators as needed
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
- Statistical analysis or aggregations (mean, sum, count, min, max)
- Visualization requests
- Queries involving relationships/links between object types
- Queries needing pagination (large result sets)

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

## AVAILABLE ACTIONS
- **fetch**: Query an object type with filters (supports rich filter operators)
- **aggregate**: Compute aggregations (mean, sum, count, min, max, median) grouped by columns
- **discover_links**: Find what object types can be linked from a given type
- **analyze**: Analyze fetched data with LLM
- **merge**: Combine data from two previous fetch steps (use step numbers as left/right references)
- **visualize**: Create a chart (only specify chart_type and title - columns will be auto-inferred)
- **answer**: Generate final response

## FILTER SYNTAX
Use rich filter operators in fetch/aggregate steps:
- Simple equality: {{"status": "completed"}}
- Comparison: {{"value": {{"$gt": 50}}}}, $gte, $lt, $lte
- Not equal: {{"status": {{"$ne": "cancelled"}}}}
- List membership: {{"surfactant": {{"$in": ["PS80", "PS20"]}}}}
- Not in list: {{"status": {{"$nin": ["cancelled", "failed"]}}}}
- Text contains: {{"name": {{"$contains": "stability"}}}}
- Starts with: {{"sample_id": {{"$startswith": "SAM00"}}}}
- Combine: {{"value": {{"$gt": 10, "$lt": 50}}}}

## MERGE ACTION
Use merge to combine data from two fetch steps:
- left: step number (e.g., 0 for step_0)
- right: step number (e.g., 1 for step_1)
- on: column name to join on (if same name in both)
- left_on / right_on: if join columns have different names
- how: inner (default), left, right, outer

## PAGINATION
For large result sets, use pagination in fetch:
- paginate: true
- offset: 0 (starting position)
- order_by: <column_name>
- order_direction: asc | desc

## VISUALIZATION NOTES
- Only specify chart_type and title in chart_spec
- The x, y, and color columns will be automatically inferred based on actual data columns
- Chart types: box (distributions), bar/grouped_bar (comparisons), line (trends), scatter (relationships)

## EXAMPLE PLANS

Example 1 - Aggregation:
```yaml
plan:
  - step: 1
    action: aggregate
    object_type: Result
    group_by: [measurement_type]
    aggregations:
      value: mean
      result_id: count
    filters: {{}}
    description: Calculate average values and count by measurement type
```

Example 2 - Link Discovery:
```yaml
plan:
  - step: 1
    action: discover_links
    object_type: Sample
    description: Find what object types are linked to Sample
```

Example 3 - Fetch with pagination:
```yaml
plan:
  - step: 1
    action: fetch
    object_type: Result
    filters:
      measurement_type: turbidity
    paginate: true
    limit: 100
    offset: 0
    order_by: value
    order_direction: desc
    description: Fetch turbidity results ordered by value
```

Example 4 - Merge data from two fetches:
```yaml
plan:
  - step: 1
    action: fetch
    object_type: Sample
    filters: {{}}
    description: Fetch all samples with surfactant info
  - step: 2
    action: fetch
    object_type: Result
    filters:
      measurement_type: turbidity
    description: Fetch turbidity results
  - step: 3
    action: merge
    left: 0
    right: 1
    on: sample_id
    how: inner
    description: Join samples with their turbidity results
  - step: 4
    action: visualize
    chart_spec:
      chart_type: box
      title: Turbidity by Surfactant
    description: Compare turbidity distributions across surfactants
```

Return the plan in YAML:
```yaml
thinking: |
    <your analysis of what needs to be done>
plan:
  - step: 1
    action: <action_type>
    <action-specific params>
    description: <what this step does>
total_steps: <number>
```
"""
