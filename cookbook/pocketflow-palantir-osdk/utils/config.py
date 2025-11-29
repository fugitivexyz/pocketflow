"""
Configuration schema for the Palantir OSDK Bot.

This module provides:
- AppConfig dataclass with nested sections for all settings
- Simple YAML loading with environment variable overrides
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


# =============================================================================
# Configuration Schema
# =============================================================================

@dataclass
class ModelSettings:
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: Optional[int] = None

@dataclass
class AgentBehavior:
    complexity_mode: str = "auto"
    max_plan_steps: int = 5
    max_query_results: int = 100
    max_retries: int = 2
    retry_delay: float = 1.0
    query_timeout: int = 120
    max_iterations: int = 10

@dataclass
class Visualization:
    default_chart_type: str = "bar"
    allowed_chart_types: List[str] = field(default_factory=lambda: [
        "bar", "grouped_bar", "line", "scatter", "box", "heatmap"
    ])
    color_theme: str = "plotly"
    auto_generate_charts: bool = True

@dataclass
class Transparency:
    show_thinking_steps: bool = True
    show_raw_prompts: bool = False
    verbosity_level: str = "medium"

@dataclass
class Connection:
    api_base_url: Optional[str] = None
    use_mock_data: bool = True
    api_key_env_var: str = "OPENAI_API_KEY"

@dataclass
class Prompts:
    coordinator_prompt: Optional[str] = None
    planner_prompt: Optional[str] = None
    simple_executor_prompt: Optional[str] = None
    answer_prompt: Optional[str] = None

@dataclass
class AppConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    agent: AgentBehavior = field(default_factory=AgentBehavior)
    visualization: Visualization = field(default_factory=Visualization)
    transparency: Transparency = field(default_factory=Transparency)
    connection: Connection = field(default_factory=Connection)
    prompts: Prompts = field(default_factory=Prompts)


# =============================================================================
# Loading Logic
# =============================================================================

def get_config_path() -> Path:
    return Path(__file__).parent.parent / "config.yaml"

def load_config() -> AppConfig:
    """Load configuration from config.yaml with env var overrides."""
    config_path = get_config_path()
    data = {}
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    # Helper to safely get nested dicts
    def get_section(name): return data.get(name, {})

    config = AppConfig(
        model=ModelSettings(**get_section("model")),
        agent=AgentBehavior(**get_section("agent")),
        visualization=Visualization(**get_section("visualization")),
        transparency=Transparency(**get_section("transparency")),
        connection=Connection(**get_section("connection")),
        prompts=Prompts(**get_section("prompts")),
    )
    
    # Environment variable overrides
    if os.environ.get("OPENAI_MODEL"):
        config.model.model = os.environ["OPENAI_MODEL"]
    
    if os.environ.get("OPENAI_TEMPERATURE"):
        try:
            config.model.temperature = float(os.environ["OPENAI_TEMPERATURE"])
        except ValueError:
            pass
            
    if os.environ.get("USE_MOCK_OSDK"):
        config.connection.use_mock_data = os.environ["USE_MOCK_OSDK"].lower() == "true"
    
    return config
