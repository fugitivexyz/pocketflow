"""
Helper utilities for PocketFlow nodes.

Provides:
- CancellationError: Exception for user-initiated cancellation
- check_cancellation: Check if execution should be cancelled
- log_thinking: Log agent thinking steps for UI transparency
- get_config_value: Safely get config values with defaults
"""

from typing import Dict, Any

try:
    from utils.streaming import log_thinking_streaming
except ImportError:
    from streaming import log_thinking_streaming


class CancellationError(Exception):
    """Raised when execution is cancelled by user."""
    pass


def check_cancellation(shared: Dict[str, Any]) -> None:
    """
    Check if cancellation has been requested.
    
    Args:
        shared: Shared state dictionary containing optional stop_event
        
    Raises:
        CancellationError: If stop_event is set
    """
    stop_event = shared.get("stop_event")
    if stop_event and stop_event.is_set():
        raise CancellationError("Execution cancelled by user")


def log_thinking(shared: Dict[str, Any], step_type: str, content: str, level: str = "medium") -> None:
    """
    Log a thinking step for UI transparency.
    
    Args:
        shared: Shared state dictionary
        step_type: Display label (e.g., "🎯 Coordinator")
        content: Content to display
        level: Verbosity level ("low", "medium", "high")
    """
    config = shared.get("config")
    if config:
        verbosity = getattr(config, 'verbosity_level', 'medium')
        levels = {"low": 1, "medium": 2, "high": 3}
        if levels.get(verbosity, 2) < levels.get(level, 2):
            return
    
    log_thinking_streaming(shared, step_type, content)


def get_config_value(shared: Dict[str, Any], attr: str, default: Any) -> Any:
    """
    Safely get config value with default.
    
    Args:
        shared: Shared state dictionary containing optional config
        attr: Attribute name to retrieve
        default: Default value if config or attr not found
        
    Returns:
        Config attribute value or default
    """
    config = shared.get("config")
    return getattr(config, attr, default) if config else default


if __name__ == "__main__":
    import threading
    
    # Test get_config_value
    print("Testing get_config_value...")
    shared = {}
    assert get_config_value(shared, "max_retries", 3) == 3
    
    class MockConfig:
        max_retries = 5
        verbosity_level = "high"
    
    shared = {"config": MockConfig()}
    assert get_config_value(shared, "max_retries", 3) == 5
    assert get_config_value(shared, "nonexistent", "default") == "default"
    print("✅ get_config_value works correctly")
    
    # Test check_cancellation
    print("\nTesting check_cancellation...")
    shared = {}
    check_cancellation(shared)  # Should not raise
    
    stop_event = threading.Event()
    shared = {"stop_event": stop_event}
    check_cancellation(shared)  # Should not raise
    
    stop_event.set()
    try:
        check_cancellation(shared)
        assert False, "Should have raised CancellationError"
    except CancellationError:
        print("✅ check_cancellation correctly raises CancellationError")
    
    print("\n✅ All helper tests passed!")
