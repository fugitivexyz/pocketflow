"""
OSDK Client abstraction for Palantir Foundry.

Provides:
- BaseOSDKClient: Abstract interface for OSDK operations
- MockOSDKClient: Mock implementation loading from data/mock_pharma.json
- PalantirOSDKClient: Stub for real Palantir OSDK integration

To use with real Palantir OSDK:
1. Generate your OSDK package using Palantir's SDK generator
2. Implement PalantirOSDKClient using your generated SDK
3. Set USE_MOCK_OSDK=false in environment

Supported Filter Operators:
- $eq: Equal to (default if no operator specified)
- $ne: Not equal to
- $gt: Greater than
- $gte: Greater than or equal
- $lt: Less than
- $lte: Less than or equal
- $in: Value in list
- $nin: Value not in list
- $contains: String contains (case-insensitive)
- $startswith: String starts with

Example filters:
    {"status": "completed"}                      # Simple equality
    {"ph": {"$gt": 6.0, "$lt": 7.5}}            # Range
    {"surfactant": {"$in": ["PS80", "PS20"]}}   # List membership
    {"name": {"$contains": "stability"}}         # Text search
"""

import os
import json
import random
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseOSDKClient(ABC):
    """Abstract base class defining the OSDK interface."""
    
    @abstractmethod
    def list_object_types(self) -> List[str]:
        """Return list of available object type names."""
        pass
    
    @abstractmethod
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        """Get schema for an object type."""
        pass
    
    @abstractmethod
    def query_objects(self, object_type: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> pd.DataFrame:
        """Query objects with optional filters (supports rich filter operators)."""
        pass
    
    @abstractmethod
    def query_objects_paginated(
        self, 
        object_type: str, 
        filters: Optional[Dict[str, Any]] = None, 
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_direction: str = "asc"
    ) -> Dict[str, Any]:
        """
        Query objects with pagination support.
        
        Args:
            object_type: The type of object to query
            filters: Optional filters (supports rich filter operators)
            limit: Maximum number of results per page
            offset: Number of results to skip
            order_by: Column to sort by
            order_direction: "asc" or "desc"
            
        Returns:
            {
                "data": pd.DataFrame,
                "total_count": int,
                "limit": int,
                "offset": int,
                "has_more": bool
            }
        """
        pass
    
    @abstractmethod
    def get_linked_objects(self, object_type: str, primary_key: str, link_type: str) -> pd.DataFrame:
        """Get objects linked to a specific object."""
        pass
    
    @abstractmethod
    def aggregate_objects(
        self,
        object_type: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Aggregate objects with grouping.
        
        Args:
            object_type: Object type to aggregate
            group_by: Columns to group by
            aggregations: {column: operation} where operation is one of:
                         mean, sum, count, min, max, median, std, first, last
            filters: Optional filters to apply before aggregation
            
        Returns:
            DataFrame with grouped aggregation results
        """
        pass
    
    @abstractmethod
    def list_link_types(self, object_type: str) -> List[Dict[str, str]]:
        """
        List available link types from an object type.
        
        Args:
            object_type: The source object type
            
        Returns:
            List of link definitions:
            [{"link_name": str, "source_field": str, "target_type": str, "target_field": str}, ...]
        """
        pass


class MockOSDKClient(BaseOSDKClient):
    """Mock OSDK client loading data from JSON file."""
    
    def __init__(self):
        self._load_data()
        self._generate_results()  # Generate results dynamically for variety
    
    def _load_data(self):
        """Load mock data from JSON file."""
        data_path = Path(__file__).parent.parent / "data" / "mock_pharma.json"
        with open(data_path, "r") as f:
            data = json.load(f)
        
        self._schemas = data["schemas"]
        self._links = data["links"]
        self._data = {
            "Protein": pd.DataFrame(data["proteins"]),
            "Excipient": pd.DataFrame(data["excipients"]),
            "Experiment": pd.DataFrame(data["experiments"]),
            "Sample": pd.DataFrame(data["samples"]),
        }
    
    def _generate_results(self):
        """Generate result measurements for samples."""
        results = []
        result_id = 1
        measurement_types = ["SEC_monomer", "SEC_aggregate", "turbidity", "viscosity", "sub_visible_particles"]
        time_points = ["D0", "D7", "D14", "D28"]
        
        for _, sample in self._data["Sample"].iterrows():
            for mtype in measurement_types:
                for tp in time_points[:random.randint(2, 4)]:
                    if mtype == "SEC_monomer":
                        value, unit = round(random.uniform(95, 99.5), 2), "%"
                    elif mtype == "SEC_aggregate":
                        value, unit = round(random.uniform(0.1, 3.0), 2), "%"
                    elif mtype == "turbidity":
                        value, unit = round(random.uniform(0.01, 0.5), 3), "OD350"
                    elif mtype == "viscosity":
                        value, unit = round(random.uniform(5, 50), 1), "cP"
                    else:
                        value, unit = random.randint(100, 10000), "particles/mL"
                    
                    results.append({
                        "result_id": f"RES{result_id:05d}",
                        "sample_id": sample["sample_id"],
                        "experiment_id": sample["experiment_id"],
                        "measurement_type": mtype,
                        "time_point": tp,
                        "value": value,
                        "unit": unit,
                        "measurement_date": "2024-02-15",
                        "analyst": random.choice(["Tech A", "Tech B", "Tech C"]),
                    })
                    result_id += 1
        
        self._data["Result"] = pd.DataFrame(results)
    
    def _apply_filter(self, df: pd.DataFrame, field: str, condition: Any) -> pd.DataFrame:
        """
        Apply a single filter condition to a DataFrame.
        
        Supports both simple values (backward compatible) and rich operators:
        - Simple: {"status": "completed"} or {"status": ["a", "b"]}
        - Rich: {"ph": {"$gt": 6.0, "$lt": 7.5}}
        """
        if field not in df.columns:
            return df
            
        # Backward compatible: simple equality or list check
        if not isinstance(condition, dict):
            if isinstance(condition, list):
                return df[df[field].isin(condition)]
            return df[df[field] == condition]
        
        # Rich filter operators
        mask = pd.Series([True] * len(df), index=df.index)
        
        for op, value in condition.items():
            if op == "$eq":
                mask &= df[field] == value
            elif op == "$ne":
                mask &= df[field] != value
            elif op == "$gt":
                mask &= df[field] > value
            elif op == "$gte":
                mask &= df[field] >= value
            elif op == "$lt":
                mask &= df[field] < value
            elif op == "$lte":
                mask &= df[field] <= value
            elif op == "$in":
                mask &= df[field].isin(value)
            elif op == "$nin":
                mask &= ~df[field].isin(value)
            elif op == "$contains":
                mask &= df[field].astype(str).str.contains(str(value), case=False, na=False)
            elif op == "$startswith":
                mask &= df[field].astype(str).str.startswith(str(value), na=False)
            # Ignore unknown operators
                
        return df[mask]
    
    def list_object_types(self) -> List[str]:
        return list(self._schemas.keys())
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        if object_type not in self._schemas:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        schema = self._schemas[object_type].copy()
        # Enhance schema with discovered link information
        schema["available_links"] = self.list_link_types(object_type)
        return schema
    
    def query_objects(self, object_type: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> pd.DataFrame:
        """Query objects with optional rich filters."""
        if object_type not in self._data:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        
        df = self._data[object_type].copy()
        if filters:
            for field, condition in filters.items():
                df = self._apply_filter(df, field, condition)
        return df.head(limit)
    
    def query_objects_paginated(
        self, 
        object_type: str, 
        filters: Optional[Dict[str, Any]] = None, 
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_direction: str = "asc"
    ) -> Dict[str, Any]:
        """Query objects with pagination support."""
        if object_type not in self._data:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        
        df = self._data[object_type].copy()
        
        # Apply filters
        if filters:
            for field, condition in filters.items():
                df = self._apply_filter(df, field, condition)
        
        total_count = len(df)
        
        # Apply ordering
        if order_by and order_by in df.columns:
            df = df.sort_values(by=order_by, ascending=(order_direction.lower() == "asc"))
        
        # Apply pagination
        paginated_df = df.iloc[offset:offset + limit].reset_index(drop=True)
        
        return {
            "data": paginated_df,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
    
    def get_linked_objects(self, object_type: str, primary_key: str, link_type: str) -> pd.DataFrame:
        link_key = f"{object_type}->{link_type}"
        if link_key not in self._links:
            raise ValueError(f"No link from {object_type} to {link_type}")
        
        source_field, target_field = self._links[link_key]
        source_df = self._data[object_type]
        pk_field = self._schemas[object_type]["primary_key"]
        source_row = source_df[source_df[pk_field] == primary_key]
        
        if source_row.empty:
            raise ValueError(f"Object not found: {object_type} with {pk_field}={primary_key}")
        
        link_value = source_row[source_field].iloc[0]
        return self._data[link_type][self._data[link_type][target_field] == link_value]
    
    def aggregate_objects(
        self,
        object_type: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Aggregate objects with grouping.
        
        Supported aggregation operations:
        - mean: Average value
        - sum: Sum of values
        - count: Number of values
        - min: Minimum value
        - max: Maximum value
        - median: Median value
        - std: Standard deviation
        - first: First value in group
        - last: Last value in group
        """
        if object_type not in self._data:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        
        # Start with filtered data (use a high limit to get all data for aggregation)
        df = self.query_objects(object_type, filters, limit=100000)
        
        if df.empty:
            return pd.DataFrame()
        
        # Validate group_by columns exist
        valid_group_by = [col for col in group_by if col in df.columns]
        if not valid_group_by:
            raise ValueError(f"No valid group_by columns found. Available: {list(df.columns)}")
        
        # Build aggregation dictionary
        agg_dict = {}
        for col, op in aggregations.items():
            if col in df.columns:
                # Map operation names to pandas aggregation functions
                op_lower = op.lower()
                if op_lower in ["mean", "sum", "count", "min", "max", "median", "std", "first", "last"]:
                    agg_dict[col] = op_lower
                elif op_lower == "avg":
                    agg_dict[col] = "mean"
                else:
                    # Default to mean for unknown operations
                    agg_dict[col] = "mean"
        
        if not agg_dict:
            raise ValueError(f"No valid aggregation columns found. Available: {list(df.columns)}")
        
        # Perform aggregation
        result = df.groupby(valid_group_by, as_index=False).agg(agg_dict)
        return result
    
    def list_link_types(self, object_type: str) -> List[Dict[str, str]]:
        """List available link types from an object type."""
        links = []
        for link_key, (source_field, target_field) in self._links.items():
            source_type, target_type = link_key.split("->")
            if source_type == object_type:
                links.append({
                    "link_name": target_type,
                    "source_field": source_field,
                    "target_type": target_type,
                    "target_field": target_field
                })
        return links


class PalantirOSDKClient(BaseOSDKClient):
    """
    Stub for real Palantir OSDK integration.
    
    Implement using your generated OSDK package:
    
        from your_osdk import FoundryClient
        
        class PalantirOSDKClient(BaseOSDKClient):
            def __init__(self):
                self.client = FoundryClient()
            
            def query_objects(self, object_type, filters=None, limit=100):
                obj_class = getattr(self.client.ontology.objects, object_type)
                query = obj_class
                if filters:
                    for field, value in filters.items():
                        # Handle rich filter operators
                        if isinstance(value, dict):
                            for op, val in value.items():
                                prop = getattr(obj_class.object_type, field)
                                if op == "$gt": query = query.where(prop > val)
                                elif op == "$lt": query = query.where(prop < val)
                                elif op == "$in": query = query.where(prop.in_(val))
                                # ... handle other operators
                        else:
                            query = query.where(getattr(obj_class, field) == value)
                results = query.take(limit)
                return pd.DataFrame([obj.dict() for obj in results])
            
            def query_objects_paginated(self, object_type, filters=None, limit=100, 
                                        offset=0, order_by=None, order_direction="asc"):
                # Use OSDK's built-in pagination
                obj_class = getattr(self.client.ontology.objects, object_type)
                # ... build query with filters and ordering
                # Use page_size and page_token for pagination
                pass
            
            def aggregate_objects(self, object_type, group_by, aggregations, filters=None):
                # Use OSDK's aggregate() method
                obj_class = getattr(self.client.ontology.objects, object_type)
                agg_spec = {name: getattr(obj_class, col).op() 
                           for col, op in aggregations.items()}
                return obj_class.aggregate(agg_spec).compute()
            
            def list_link_types(self, object_type):
                # Use OSDK's outgoing link types endpoint
                pass
    """
    
    def __init__(self):
        raise NotImplementedError("Implement PalantirOSDKClient with your generated OSDK")
    
    def list_object_types(self) -> List[str]:
        raise NotImplementedError()
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def query_objects(self, object_type: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> pd.DataFrame:
        raise NotImplementedError()
    
    def query_objects_paginated(
        self, 
        object_type: str, 
        filters: Optional[Dict[str, Any]] = None, 
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_direction: str = "asc"
    ) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def get_linked_objects(self, object_type: str, primary_key: str, link_type: str) -> pd.DataFrame:
        raise NotImplementedError()
    
    def aggregate_objects(
        self,
        object_type: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        raise NotImplementedError()
    
    def list_link_types(self, object_type: str) -> List[Dict[str, str]]:
        raise NotImplementedError()


def get_osdk_client() -> BaseOSDKClient:
    """Factory to get appropriate OSDK client. Set USE_MOCK_OSDK=false for real client."""
    if os.environ.get("USE_MOCK_OSDK", "true").lower() == "true":
        return MockOSDKClient()
    return PalantirOSDKClient()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MockOSDKClient")
    print("=" * 60)
    client = MockOSDKClient()
    
    print("\n## Object types:", client.list_object_types())
    
    # Basic query
    print("\n## Experiments:", client.query_objects("Experiment"))
    
    # Simple filter (backward compatible)
    print("\n## Completed experiments:", client.query_objects("Experiment", {"status": "completed"}))
    
    # Rich filters
    print("\n## Rich filter - Samples with PS80 or PS20:")
    print(client.query_objects("Sample", {"surfactant": {"$in": ["Polysorbate 80", "Polysorbate 20"]}}, limit=5))
    
    print("\n## Rich filter - Results with value > 50:")
    print(client.query_objects("Result", {"value": {"$gt": 50}}, limit=5))
    
    print("\n## Rich filter - Combined conditions (turbidity results > 0.1 and < 0.3):")
    print(client.query_objects("Result", {
        "measurement_type": "turbidity",
        "value": {"$gt": 0.1, "$lt": 0.3}
    }, limit=5))
    
    # Pagination
    print("\n## Pagination - Page 1 (5 samples):")
    page1 = client.query_objects_paginated("Sample", limit=5, offset=0)
    print(f"   Data: {len(page1['data'])} rows, Total: {page1['total_count']}, Has more: {page1['has_more']}")
    
    print("\n## Pagination - Page 2 (5 samples):")
    page2 = client.query_objects_paginated("Sample", limit=5, offset=5)
    print(f"   Data: {len(page2['data'])} rows, Total: {page2['total_count']}, Has more: {page2['has_more']}")
    
    print("\n## Pagination with ordering:")
    ordered = client.query_objects_paginated("Result", limit=5, order_by="value", order_direction="desc")
    print(ordered["data"][["result_id", "value"]])
    
    # Aggregations
    print("\n## Aggregation - Average value by measurement type:")
    agg = client.aggregate_objects(
        "Result",
        group_by=["measurement_type"],
        aggregations={"value": "mean", "result_id": "count"}
    )
    print(agg)
    
    print("\n## Aggregation - Min/max value by surfactant and measurement:")
    agg2 = client.aggregate_objects(
        "Result",
        group_by=["measurement_type"],
        aggregations={"value": "min"},
        filters={"measurement_type": {"$in": ["turbidity", "viscosity"]}}
    )
    print(agg2)
    
    # Link discovery
    print("\n## Link discovery - Links from Sample:")
    links = client.list_link_types("Sample")
    for link in links:
        print(f"   {link['link_name']}: {link['source_field']} -> {link['target_type']}.{link['target_field']}")
    
    print("\n## Link discovery - Links from Experiment:")
    links = client.list_link_types("Experiment")
    for link in links:
        print(f"   {link['link_name']}: {link['source_field']} -> {link['target_type']}.{link['target_field']}")
    
    # Linked objects
    print("\n## Linked objects - Results for SAM0001:")
    print(client.get_linked_objects("Sample", "SAM0001", "Result").head(3))
    
    # Schema with links
    print("\n## Enhanced schema (with available_links):")
    schema = client.get_object_schema("Sample")
    print(f"   Primary key: {schema['primary_key']}")
    print(f"   Available links: {schema['available_links']}")
