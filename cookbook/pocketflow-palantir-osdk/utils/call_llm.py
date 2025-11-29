"""
LLM utility functions for OpenAI and OpenAI-compatible providers.

Supports:
- OpenAI (default)
- Azure OpenAI (via OPENAI_BASE_URL)
- Local models (via OPENAI_BASE_URL, e.g., Ollama, vLLM)

Environment Variables:
- OPENAI_API_KEY: Your API key
- OPENAI_BASE_URL: (Optional) Custom base URL
- OPENAI_MODEL: (Optional) Model name, defaults to "gpt-4o"
"""

import os
import re
import yaml
from openai import OpenAI
from typing import Optional


def get_client(base_url: Optional[str] = None) -> OpenAI:
    """Get OpenAI client configured from environment."""
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[list] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call the LLM with a prompt and optional conversation history.

    Args:
        prompt: The user's current message/prompt
        system_prompt: Optional system message to set context
        history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]
        model: Optional model override (defaults to OPENAI_MODEL env var or "gpt-4o")
        temperature: Optional temperature override (defaults to 0.1)

    Returns:
        The LLM's response as a string
    """
    client = get_client()
    
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    temperature = temperature if temperature is not None else float(os.environ.get("OPENAI_TEMPERATURE", "0.1"))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def extract_yaml(text: str) -> dict:
    """
    Extract and parse YAML from LLM response.
    
    Handles responses with ```yaml ... ``` code blocks or raw YAML.
    """
    # Try to extract from code block first
    yaml_match = re.search(r"```ya?ml\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    
    if yaml_match:
        yaml_str = yaml_match.group(1).strip()
    else:
        # Try to find YAML-like content
        lines = text.strip().split('\n')
        yaml_lines = []
        in_yaml = False
        
        for line in lines:
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:', line) or line.strip().startswith('-'):
                in_yaml = True
            if in_yaml:
                yaml_lines.append(line)
        
        yaml_str = '\n'.join(yaml_lines) if yaml_lines else text.strip()
    
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML: {e}\n\nOriginal text:\n{text}")


if __name__ == "__main__":
    print("## Testing call_llm")
    response = call_llm("In a few words, what is 2+2?")
    print(f"Response: {response}")
    
    print("\n## Testing extract_yaml")
    test = """Here's my analysis:
```yaml
action: query
object_type: Experiment
filters:
  status: completed
```
"""
    print(f"Parsed: {extract_yaml(test)}")
