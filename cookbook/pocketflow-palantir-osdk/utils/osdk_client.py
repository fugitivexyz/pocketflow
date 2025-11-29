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
"""

import os
import json
import random
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        """Query objects with optional filters."""
        pass
    
    @abstractmethod
    def get_linked_objects(self, object_type: str, primary_key: str, link_type: str) -> pd.DataFrame:
        """Get objects linked to a specific object."""
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
    
    def list_object_types(self) -> List[str]:
        return list(self._schemas.keys())
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        if object_type not in self._schemas:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        return self._schemas[object_type]
    
    def query_objects(self, object_type: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> pd.DataFrame:
        if object_type not in self._data:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        
        df = self._data[object_type].copy()
        if filters:
            for field, value in filters.items():
                if field in df.columns:
                    df = df[df[field].isin(value)] if isinstance(value, list) else df[df[field] == value]
        return df.head(limit)
    
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
                        query = query.where(getattr(obj_class, field) == value)
                results = query.take(limit)
                return pd.DataFrame([obj.dict() for obj in results])
    """
    
    def __init__(self):
        raise NotImplementedError("Implement PalantirOSDKClient with your generated OSDK")
    
    def list_object_types(self) -> List[str]:
        raise NotImplementedError()
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def query_objects(self, object_type: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> pd.DataFrame:
        raise NotImplementedError()
    
    def get_linked_objects(self, object_type: str, primary_key: str, link_type: str) -> pd.DataFrame:
        raise NotImplementedError()


def get_osdk_client() -> BaseOSDKClient:
    """Factory to get appropriate OSDK client. Set USE_MOCK_OSDK=false for real client."""
    if os.environ.get("USE_MOCK_OSDK", "true").lower() == "true":
        return MockOSDKClient()
    return PalantirOSDKClient()


if __name__ == "__main__":
    print("## Testing MockOSDKClient")
    client = MockOSDKClient()
    
    print("\n## Object types:", client.list_object_types())
    print("\n## Experiments:", client.query_objects("Experiment"))
    print("\n## Completed:", client.query_objects("Experiment", {"status": "completed"}))
    print("\n## Samples (PS80):", client.query_objects("Sample", {"surfactant": "Polysorbate 80"}, limit=5))
    print("\n## Results for SAM0001:", client.get_linked_objects("Sample", "SAM0001", "Result"))
