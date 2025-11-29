"""
Simple configuration using environment variables with sensible defaults.

Environment Variables:
- OPENAI_API_KEY: Your API key (required)
- OPENAI_BASE_URL: Custom endpoint for Azure/local models (optional)
- OPENAI_MODEL: Model name (default: gpt-4o)
- OPENAI_TEMPERATURE: Temperature (default: 0.1)
- USE_MOCK_OSDK: Use mock data (default: true)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class Transparency:
    show_thinking_steps: bool = True
    show_raw_prompts: bool = False
    verbosity_level: str = "medium"


@dataclass
class Connection:
    api_base_url: Optional[str] = None
    use_mock_data: bool = True


@dataclass
class Config:
    """Configuration with nested sections for UI compatibility."""
    model: ModelSettings = field(default_factory=ModelSettings)
    agent: AgentBehavior = field(default_factory=AgentBehavior)
    transparency: Transparency = field(default_factory=Transparency)
    connection: Connection = field(default_factory=Connection)
    color_theme: str = "plotly_white"
    
    # Flat access for node usage
    @property
    def max_retries(self): return self.agent.max_retries
    @property
    def retry_delay(self): return self.agent.retry_delay
    @property
    def max_plan_steps(self): return self.agent.max_plan_steps
    @property
    def max_query_results(self): return self.agent.max_query_results
    @property
    def verbosity_level(self): return self.transparency.verbosity_level


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        model=ModelSettings(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ["OPENAI_MAX_TOKENS"]) if os.environ.get("OPENAI_MAX_TOKENS") else None,
        ),
        agent=AgentBehavior(
            max_retries=int(os.environ.get("MAX_RETRIES", "2")),
            retry_delay=float(os.environ.get("RETRY_DELAY", "1.0")),
            max_plan_steps=int(os.environ.get("MAX_PLAN_STEPS", "5")),
            max_query_results=int(os.environ.get("MAX_QUERY_RESULTS", "100")),
            query_timeout=int(os.environ.get("QUERY_TIMEOUT", "120")),
        ),
        transparency=Transparency(
            verbosity_level=os.environ.get("VERBOSITY_LEVEL", "medium"),
        ),
        connection=Connection(
            api_base_url=os.environ.get("OPENAI_BASE_URL"),
            use_mock_data=os.environ.get("USE_MOCK_OSDK", "true").lower() == "true",
        ),
        color_theme=os.environ.get("CHART_THEME", "plotly_white"),
    )


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


if __name__ == "__main__":
    config = load_config()
    print(f"Model: {config.model.model}")
    print(f"Temperature: {config.model.temperature}")
    print(f"Max retries: {config.max_retries}")
    print(f"Verbosity: {config.verbosity_level}")
