"""
LLM utility functions for OpenAI and OpenAI-compatible providers.

Supports:
- OpenAI (default)
- Azure OpenAI (via OPENAI_BASE_URL)
- Local models (via OPENAI_BASE_URL, e.g., Ollama, vLLM)
- Palantir AIP (if OpenAI-compatible endpoint available)

Environment Variables:
- OPENAI_API_KEY: Your API key
- OPENAI_BASE_URL: (Optional) Custom base URL for compatible providers
- OPENAI_MODEL: (Optional) Model name, defaults to "gpt-4o"

Config Override:
- When an AppConfig is passed, its settings take precedence over env vars
"""

import os
import re
import yaml
from openai import OpenAI
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.config import AppConfig


def get_client(config: Optional["AppConfig"] = None):
    """
    Get OpenAI client configured from config or environment variables.
    
    Args:
        config: Optional AppConfig for settings override
    """
    # Determine base URL
    base_url = None
    if config and config.connection.api_base_url:
        base_url = config.connection.api_base_url
    else:
        base_url = os.environ.get("OPENAI_BASE_URL")
    
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
        base_url=base_url,
    )


def call_llm(
    prompt: str,
    system_prompt: str = None,
    history: list = None,
    model: str = None,
    temperature: float = None,
    config: Optional["AppConfig"] = None,
    max_tokens: int = None,
    top_p: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None,
) -> str:
    """
    Call the LLM with a prompt and optional conversation history.

    Args:
        prompt: The user's current message/prompt
        system_prompt: Optional system message to set context
        history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]
        model: Optional model override (defaults to config or OPENAI_MODEL env var or "gpt-4o")
        temperature: Optional temperature override (defaults to config or OPENAI_TEMPERATURE env var or 0.1)
        config: Optional AppConfig for settings (takes precedence over env vars)
        max_tokens: Optional max tokens override
        top_p: Optional top_p override
        frequency_penalty: Optional frequency penalty override
        presence_penalty: Optional presence penalty override

    Returns:
        The LLM's response as a string
    """
    client = get_client(config)
    
    # Resolve parameters with priority: explicit arg > config > env var > default
    if model is None:
        if config:
            model = config.model.model
        else:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    
    if temperature is None:
        if config:
            temperature = config.model.temperature
        else:
            temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.1"))
    
    if max_tokens is None and config:
        max_tokens = config.model.max_tokens

    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add conversation history if provided
    if history:
        messages.extend(history)

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    # Build API call kwargs
    api_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    # Add optional parameters if set
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens
    if top_p is not None and top_p != 1.0:  # Only add if non-default
        api_kwargs["top_p"] = top_p
    if frequency_penalty is not None and frequency_penalty != 0.0:
        api_kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None and presence_penalty != 0.0:
        api_kwargs["presence_penalty"] = presence_penalty

    response = client.chat.completions.create(**api_kwargs)

    return response.choices[0].message.content


def extract_yaml(text: str) -> dict:
    """
    Extract and parse YAML from LLM response.
    
    Handles responses with ```yaml ... ``` code blocks or raw YAML.
    
    Args:
        text: The LLM response text
        
    Returns:
        Parsed YAML as a dictionary
        
    Raises:
        ValueError: If no valid YAML found
    """
    # Try to extract from code block first
    yaml_match = re.search(r"```ya?ml\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    
    if yaml_match:
        yaml_str = yaml_match.group(1).strip()
    else:
        # Try to find YAML-like content (starts with key: or -)
        lines = text.strip().split('\n')
        yaml_lines = []
        in_yaml = False
        
        for line in lines:
            # Detect start of YAML content
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:', line) or line.strip().startswith('-'):
                in_yaml = True
            
            if in_yaml:
                yaml_lines.append(line)
        
        yaml_str = '\n'.join(yaml_lines) if yaml_lines else text.strip()
    
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML: {e}\n\nOriginal text:\n{text}")


def format_history_for_prompt(history: list, max_turns: int = 10) -> str:
    """
    Format conversation history into a string for inclusion in prompts.
    
    Args:
        history: List of message dicts with 'role' and 'content'
        max_turns: Maximum number of turns to include (default 10)
        
    Returns:
        Formatted string representation of history
    """
    if not history:
        return "No previous conversation."
    
    # Take last N turns
    recent = history[-max_turns * 2:]  # *2 because each turn has user + assistant
    
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)


# Test the utilities
if __name__ == "__main__":
    print("## Testing call_llm")
    prompt = "In a few words, what is the meaning of life?"
    print(f"## Prompt: {prompt}")
    response = call_llm(prompt)
    print(f"## Response: {response}")
    
    print("\n## Testing extract_yaml")
    test_response = """
Here's my analysis:

```yaml
action: query_osdk
object_type: Experiment
filters:
  status: completed
reason: Need to find completed experiments
```

Let me know if you need more details.
"""
    parsed = extract_yaml(test_response)
    print(f"## Parsed YAML: {parsed}")
