"""
Streaming utilities for real-time UI updates during flow execution.

Uses a callback pattern with threading to allow PocketFlow to run
in the background while Streamlit updates the UI in real-time.
"""

import threading
import queue
import contextvars
from typing import Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum


class StepType(Enum):
    """Types of thinking steps for UI rendering."""
    THINKING = "thinking"
    DECISION = "decision"
    ACTION = "action"
    DATA = "data"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class ThinkingStep:
    """A single thinking step to be displayed in the UI."""
    step_type: str
    content: str
    metadata: Optional[dict] = None


class StreamingCallback:
    """
    Callback handler for streaming thinking steps to the UI.
    
    This class provides a thread-safe way to communicate between
    the PocketFlow execution thread and the Streamlit UI thread.
    
    Usage:
        callback = StreamingCallback()
        
        # In flow execution (background thread):
        callback.on_step("🤔 Decision", "Analyzing query...")
        
        # In Streamlit UI (main thread):
        for step in callback.iter_steps():
            st.write(step.content)
    """
    
    def __init__(self):
        self._queue: queue.Queue[Optional[ThinkingStep]] = queue.Queue()
        self._complete = threading.Event()
        self._error: Optional[Exception] = None
        self._result: Optional[dict] = None
    
    def on_step(self, step_type: str, content: str, metadata: dict = None):
        """
        Called by nodes to report a thinking step.
        
        Args:
            step_type: Display label (e.g., "🤔 Decision", "🔍 Query")
            content: The content to display
            metadata: Optional additional data
        """
        step = ThinkingStep(step_type=step_type, content=content, metadata=metadata)
        self._queue.put(step)
    
    def on_complete(self, result: dict):
        """Called when the flow completes successfully."""
        self._result = result
        self._queue.put(None)  # Signal completion
        self._complete.set()
    
    def on_error(self, error: Exception):
        """Called when the flow encounters an error."""
        self._error = error
        self._queue.put(None)  # Signal completion
        self._complete.set()
    
    def iter_steps(self, timeout: float = 0.1):
        """
        Iterate over thinking steps as they arrive.
        
        Yields ThinkingStep objects until the flow completes.
        Use a small timeout to allow Streamlit to update the UI.
        
        Args:
            timeout: How long to wait for each step (seconds)
            
        Yields:
            ThinkingStep objects as they arrive
        """
        while not self._complete.is_set():
            try:
                step = self._queue.get(timeout=timeout)
                if step is None:
                    break
                yield step
            except queue.Empty:
                continue
        
        # Drain any remaining items
        while True:
            try:
                step = self._queue.get_nowait()
                if step is None:
                    break
                yield step
            except queue.Empty:
                break
    
    def get_result(self) -> dict:
        """Get the final result after completion."""
        if self._error:
            raise self._error
        return self._result or {}
    
    def is_complete(self) -> bool:
        """Check if the flow has completed."""
        return self._complete.is_set()


# Thread-safe callback storage using contextvars
_current_callback: contextvars.ContextVar[Optional[StreamingCallback]] = contextvars.ContextVar(
    '_current_callback', default=None
)


def set_streaming_callback(callback: Optional[StreamingCallback]):
    """Set the current streaming callback (thread-safe via contextvars)."""
    _current_callback.set(callback)


def get_streaming_callback() -> Optional[StreamingCallback]:
    """Get the current streaming callback (thread-safe via contextvars)."""
    return _current_callback.get()


def log_thinking_streaming(shared: dict, step_type: str, content: str):
    """
    Log a thinking step with optional streaming support.
    
    If a streaming callback is set, the step is sent immediately to the UI.
    The step is also always stored in shared["thinking_steps"] for persistence.
    
    Args:
        shared: The shared state dictionary
        step_type: Display label for the step
        content: The content to display
    """
    # Always store in shared for persistence
    if "thinking_steps" not in shared:
        shared["thinking_steps"] = []
    shared["thinking_steps"].append({
        "type": step_type,
        "content": content,
    })
    
    # If streaming callback is set, notify it
    callback = get_streaming_callback()
    if callback:
        callback.on_step(step_type, content)


def run_flow_with_streaming(
    flow_runner: Callable,
    callback: StreamingCallback,
    *args,
    **kwargs
) -> threading.Thread:
    """
    Run a flow in a background thread with streaming callback.
    
    Args:
        flow_runner: Function that runs the flow (e.g., run_multi_agent_query)
        callback: StreamingCallback instance for updates
        *args, **kwargs: Arguments to pass to flow_runner
        
    Returns:
        The background thread (already started)
    """
    def _run():
        try:
            set_streaming_callback(callback)
            result = flow_runner(*args, **kwargs)
            callback.on_complete(result)
        except Exception as e:
            callback.on_error(e)
        finally:
            set_streaming_callback(None)
    
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
