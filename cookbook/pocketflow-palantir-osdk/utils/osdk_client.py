"""
OSDK Client abstraction for Palantir Foundry.

This module provides:
1. BaseOSDKClient - Abstract interface for OSDK operations
2. MockOSDKClient - Mock implementation with pharma R&D sample data
3. PalantirOSDKClient - Stub for integrating real Palantir OSDK

To use with your real Palantir OSDK:
1. Generate your OSDK package using Palantir's SDK generator
2. Implement PalantirOSDKClient using your generated SDK
3. Set USE_MOCK_OSDK=false in environment

The mock data simulates a pharmaceutical R&D domain with:
- Experiment: Research experiments with protocols
- Sample: Samples tested in experiments
- Result: Measurement results from tests
- Protein: Proteins being studied
- Excipient: Excipients used in formulations
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random


class BaseOSDKClient(ABC):
    """
    Abstract base class defining the OSDK interface.
    
    Implement this interface to connect to your Palantir Foundry instance.
    """
    
    @abstractmethod
    def list_object_types(self) -> List[str]:
        """Return list of available object type names."""
        pass
    
    @abstractmethod
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        """
        Get schema for an object type.
        
        Returns:
            Dict with 'properties' (field name -> type) and 'description'
        """
        pass
    
    @abstractmethod
    def query_objects(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Query objects of a given type with optional filters.
        
        Args:
            object_type: Name of the object type to query
            filters: Dict of field_name -> value for filtering
            limit: Maximum number of results
            
        Returns:
            DataFrame with query results
        """
        pass
    
    @abstractmethod
    def get_linked_objects(
        self,
        object_type: str,
        primary_key: str,
        link_type: str,
    ) -> pd.DataFrame:
        """
        Get objects linked to a specific object.
        
        Args:
            object_type: Source object type
            primary_key: Primary key of source object
            link_type: Name of the link/relationship to traverse
            
        Returns:
            DataFrame with linked objects
        """
        pass


class MockOSDKClient(BaseOSDKClient):
    """
    Mock OSDK client with pharmaceutical R&D sample data.
    
    Use this for development and testing without a real Palantir instance.
    The data models common pharma R&D workflows:
    - Experiments studying protein stability
    - Samples with different formulations (excipients, concentrations)
    - Results from various assays (SEC, turbidity, viscosity)
    """
    
    def __init__(self):
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Generate realistic pharma R&D mock data."""
        
        # Proteins being studied
        self._proteins = pd.DataFrame([
            {"protein_id": "PROT001", "name": "Reslizumab", "molecular_weight": 147000, "type": "monoclonal_antibody"},
            {"protein_id": "PROT002", "name": "Adalimumab", "molecular_weight": 148000, "type": "monoclonal_antibody"},
            {"protein_id": "PROT003", "name": "Bevacizumab", "molecular_weight": 149000, "type": "monoclonal_antibody"},
        ])
        
        # Excipients used in formulations
        self._excipients = pd.DataFrame([
            {"excipient_id": "EXC001", "name": "Polysorbate 80", "category": "surfactant", "supplier": "Sigma"},
            {"excipient_id": "EXC002", "name": "Polysorbate 20", "category": "surfactant", "supplier": "Sigma"},
            {"excipient_id": "EXC003", "name": "Poloxamer 188", "category": "surfactant", "supplier": "BASF"},
            {"excipient_id": "EXC004", "name": "Histidine", "category": "buffer", "supplier": "Ajinomoto"},
            {"excipient_id": "EXC005", "name": "Sucrose", "category": "stabilizer", "supplier": "Pfanstiehl"},
            {"excipient_id": "EXC006", "name": "Arginine", "category": "stabilizer", "supplier": "Ajinomoto"},
            {"excipient_id": "EXC007", "name": "Sodium Chloride", "category": "tonicity_agent", "supplier": "Sigma"},
        ])
        
        # Experiments
        self._experiments = pd.DataFrame([
            {"experiment_id": "EXP001", "name": "Surfactant Screening Study", "protein_id": "PROT001", 
             "status": "completed", "start_date": "2024-01-15", "end_date": "2024-02-28", "lead_scientist": "Dr. Smith"},
            {"experiment_id": "EXP002", "name": "pH Stability Study", "protein_id": "PROT001",
             "status": "completed", "start_date": "2024-02-01", "end_date": "2024-03-15", "lead_scientist": "Dr. Johnson"},
            {"experiment_id": "EXP003", "name": "Viscosity Optimization", "protein_id": "PROT002",
             "status": "in_progress", "start_date": "2024-03-01", "end_date": None, "lead_scientist": "Dr. Williams"},
            {"experiment_id": "EXP004", "name": "Thermal Stress Study", "protein_id": "PROT001",
             "status": "completed", "start_date": "2024-01-20", "end_date": "2024-03-01", "lead_scientist": "Dr. Smith"},
            {"experiment_id": "EXP005", "name": "Freeze-Thaw Stability", "protein_id": "PROT003",
             "status": "planned", "start_date": "2024-04-01", "end_date": None, "lead_scientist": "Dr. Brown"},
        ])
        
        # Samples - various formulations tested
        samples_data = []
        sample_id = 1
        surfactants = ["Polysorbate 80", "Polysorbate 20", "Poloxamer 188"]
        concentrations = [0.01, 0.02, 0.05, 0.1]
        donors = ["male", "female"]
        stress_conditions = ["none", "pH11", "shaker", "thermal_40C", "freeze_thaw"]
        
        for exp_id in ["EXP001", "EXP002", "EXP004"]:
            for surfactant in surfactants:
                for conc in concentrations[:3]:  # Vary concentrations per experiment
                    for donor in donors:
                        samples_data.append({
                            "sample_id": f"SAM{sample_id:04d}",
                            "experiment_id": exp_id,
                            "surfactant": surfactant,
                            "surfactant_concentration": conc,
                            "donor_type": donor,
                            "stress_condition": random.choice(stress_conditions),
                            "protein_concentration": 50.0,  # mg/mL
                            "ph": round(random.uniform(5.5, 7.5), 1),
                            "created_date": "2024-02-01",
                        })
                        sample_id += 1
        
        self._samples = pd.DataFrame(samples_data)
        
        # Results - measurements from various assays
        results_data = []
        result_id = 1
        measurement_types = ["SEC_monomer", "SEC_aggregate", "turbidity", "viscosity", "sub_visible_particles"]
        time_points = ["D0", "D7", "D14", "D28"]
        
        for _, sample in self._samples.iterrows():
            for measurement_type in measurement_types:
                for time_point in time_points[:random.randint(2, 4)]:  # Not all samples have all time points
                    # Generate realistic values based on measurement type
                    if measurement_type == "SEC_monomer":
                        value = round(random.uniform(95, 99.5), 2)
                        unit = "%"
                    elif measurement_type == "SEC_aggregate":
                        value = round(random.uniform(0.1, 3.0), 2)
                        unit = "%"
                    elif measurement_type == "turbidity":
                        value = round(random.uniform(0.01, 0.5), 3)
                        unit = "OD350"
                    elif measurement_type == "viscosity":
                        value = round(random.uniform(5, 50), 1)
                        unit = "cP"
                    else:  # sub_visible_particles
                        value = random.randint(100, 10000)
                        unit = "particles/mL"
                    
                    results_data.append({
                        "result_id": f"RES{result_id:05d}",
                        "sample_id": sample["sample_id"],
                        "experiment_id": sample["experiment_id"],
                        "measurement_type": measurement_type,
                        "time_point": time_point,
                        "value": value,
                        "unit": unit,
                        "measurement_date": "2024-02-15",
                        "analyst": random.choice(["Tech A", "Tech B", "Tech C"]),
                    })
                    result_id += 1
        
        self._results = pd.DataFrame(results_data)
        
        # Define schemas
        self._schemas = {
            "Protein": {
                "description": "Proteins being studied in R&D experiments",
                "properties": {
                    "protein_id": "string (primary key)",
                    "name": "string",
                    "molecular_weight": "number (Daltons)",
                    "type": "string (monoclonal_antibody, enzyme, etc.)",
                },
                "primary_key": "protein_id",
            },
            "Excipient": {
                "description": "Excipients used in pharmaceutical formulations",
                "properties": {
                    "excipient_id": "string (primary key)",
                    "name": "string",
                    "category": "string (surfactant, buffer, stabilizer, tonicity_agent)",
                    "supplier": "string",
                },
                "primary_key": "excipient_id",
            },
            "Experiment": {
                "description": "Research experiments studying protein stability and formulation",
                "properties": {
                    "experiment_id": "string (primary key)",
                    "name": "string",
                    "protein_id": "string (foreign key to Protein)",
                    "status": "string (planned, in_progress, completed)",
                    "start_date": "date",
                    "end_date": "date (nullable)",
                    "lead_scientist": "string",
                },
                "primary_key": "experiment_id",
                "links": ["Protein", "Sample"],
            },
            "Sample": {
                "description": "Samples with specific formulations tested in experiments",
                "properties": {
                    "sample_id": "string (primary key)",
                    "experiment_id": "string (foreign key to Experiment)",
                    "surfactant": "string",
                    "surfactant_concentration": "number (%)",
                    "donor_type": "string (male, female)",
                    "stress_condition": "string (none, pH11, shaker, thermal_40C, freeze_thaw)",
                    "protein_concentration": "number (mg/mL)",
                    "ph": "number",
                    "created_date": "date",
                },
                "primary_key": "sample_id",
                "links": ["Experiment", "Result"],
            },
            "Result": {
                "description": "Measurement results from analytical tests",
                "properties": {
                    "result_id": "string (primary key)",
                    "sample_id": "string (foreign key to Sample)",
                    "experiment_id": "string (foreign key to Experiment)",
                    "measurement_type": "string (SEC_monomer, SEC_aggregate, turbidity, viscosity, sub_visible_particles)",
                    "time_point": "string (D0, D7, D14, D28)",
                    "value": "number",
                    "unit": "string",
                    "measurement_date": "date",
                    "analyst": "string",
                },
                "primary_key": "result_id",
                "links": ["Sample", "Experiment"],
            },
        }
        
        # Map object types to dataframes
        self._data = {
            "Protein": self._proteins,
            "Excipient": self._excipients,
            "Experiment": self._experiments,
            "Sample": self._samples,
            "Result": self._results,
        }
    
    def list_object_types(self) -> List[str]:
        """Return list of available object type names."""
        return list(self._schemas.keys())
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        """Get schema for an object type."""
        if object_type not in self._schemas:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        return self._schemas[object_type]
    
    def query_objects(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query objects with optional filters."""
        if object_type not in self._data:
            raise ValueError(f"Unknown object type: {object_type}. Available: {self.list_object_types()}")
        
        df = self._data[object_type].copy()
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if field in df.columns:
                    if isinstance(value, list):
                        df = df[df[field].isin(value)]
                    else:
                        df = df[df[field] == value]
        
        return df.head(limit)
    
    def get_linked_objects(
        self,
        object_type: str,
        primary_key: str,
        link_type: str,
    ) -> pd.DataFrame:
        """Get objects linked to a specific object."""
        # Define link relationships
        links = {
            ("Experiment", "Sample"): ("experiment_id", "experiment_id"),
            ("Experiment", "Protein"): ("protein_id", "protein_id"),
            ("Sample", "Result"): ("sample_id", "sample_id"),
            ("Sample", "Experiment"): ("experiment_id", "experiment_id"),
            ("Result", "Sample"): ("sample_id", "sample_id"),
            ("Result", "Experiment"): ("experiment_id", "experiment_id"),
        }
        
        link_key = (object_type, link_type)
        if link_key not in links:
            raise ValueError(f"No link from {object_type} to {link_type}")
        
        source_field, target_field = links[link_key]
        
        # Get the source object
        source_df = self._data[object_type]
        pk_field = self._schemas[object_type]["primary_key"]
        source_row = source_df[source_df[pk_field] == primary_key]
        
        if source_row.empty:
            raise ValueError(f"Object not found: {object_type} with {pk_field}={primary_key}")
        
        # Get linked value
        link_value = source_row[source_field].iloc[0]
        
        # Query target objects
        target_df = self._data[link_type]
        return target_df[target_df[target_field] == link_value]


class PalantirOSDKClient(BaseOSDKClient):
    """
    Real Palantir OSDK client implementation.
    
    TODO: Implement this class using your generated Palantir OSDK.
    
    Example integration with a generated OSDK:
    
    ```python
    from your_generated_osdk import FoundryClient
    from your_generated_osdk.ontology.objects import Experiment, Sample, Result
    
    class PalantirOSDKClient(BaseOSDKClient):
        def __init__(self):
            # Initialize with your auth method
            self.client = FoundryClient()
        
        def query_objects(self, object_type, filters=None, limit=100):
            obj_class = getattr(self.client.ontology.objects, object_type)
            query = obj_class
            
            if filters:
                for field, value in filters.items():
                    query = query.where(getattr(obj_class, field) == value)
            
            results = query.take(limit)
            return pd.DataFrame([obj.dict() for obj in results])
    ```
    """
    
    def __init__(self):
        # TODO: Initialize your Palantir client here
        # from your_osdk_package import FoundryClient
        # self.client = FoundryClient()
        raise NotImplementedError(
            "PalantirOSDKClient is a stub. Implement it with your generated OSDK, "
            "or use MockOSDKClient for development."
        )
    
    def list_object_types(self) -> List[str]:
        # TODO: Return list from your ontology
        # return [obj.__name__ for obj in self.client.ontology.objects]
        raise NotImplementedError()
    
    def get_object_schema(self, object_type: str) -> Dict[str, Any]:
        # TODO: Extract schema from your OSDK objects
        raise NotImplementedError()
    
    def query_objects(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        # TODO: Implement using your OSDK
        raise NotImplementedError()
    
    def get_linked_objects(
        self,
        object_type: str,
        primary_key: str,
        link_type: str,
    ) -> pd.DataFrame:
        # TODO: Implement using your OSDK's link traversal
        raise NotImplementedError()


def get_osdk_client() -> BaseOSDKClient:
    """
    Factory function to get the appropriate OSDK client.
    
    Set USE_MOCK_OSDK=false to use real Palantir client.
    """
    use_mock = os.environ.get("USE_MOCK_OSDK", "true").lower() == "true"
    
    if use_mock:
        return MockOSDKClient()
    else:
        return PalantirOSDKClient()


# Test the mock client
if __name__ == "__main__":
    print("## Testing MockOSDKClient")
    client = MockOSDKClient()
    
    print("\n## Available object types:")
    print(client.list_object_types())
    
    print("\n## Experiment schema:")
    print(client.get_object_schema("Experiment"))
    
    print("\n## Query all experiments:")
    experiments = client.query_objects("Experiment")
    print(experiments)
    
    print("\n## Query completed experiments:")
    completed = client.query_objects("Experiment", filters={"status": "completed"})
    print(completed)
    
    print("\n## Query samples with Polysorbate 80:")
    samples = client.query_objects("Sample", filters={"surfactant": "Polysorbate 80"}, limit=5)
    print(samples)
    
    print("\n## Get results linked to sample SAM0001:")
    results = client.get_linked_objects("Sample", "SAM0001", "Result")
    print(results)
