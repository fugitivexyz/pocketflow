"""
CLI entry point for the Palantir OSDK Bot.

Usage:
    python main.py                              # Interactive mode
    python main.py "Show me all experiments"    # Single query mode

Examples:
    python main.py "What object types are available?"
    python main.py "Show me completed experiments"
    python main.py "Compare turbidity across surfactants"
"""

import sys
import argparse
from flow import run_query as run_multi_agent_query


def print_divider(char="=", width=70):
    print(char * width)


def print_thinking_steps(steps):
    """Display thinking steps in a readable format."""
    print_divider()
    print("🧠 AGENT THINKING")
    print_divider()
    
    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}] {step['type']}")
        print("-" * 50)
        content = step["content"]
        # Truncate very long content
        if len(content) > 1000:
            print(content[:1000])
            print(f"\n... (truncated, {len(content) - 1000} more chars)")
        else:
            print(content)


def print_result(result):
    """Display the query result."""
    # Show thinking steps
    if result.get("thinking_steps"):
        print_thinking_steps(result["thinking_steps"])
    
    # Show final answer
    print_divider()
    print("💬 ANSWER")
    print_divider()
    print(result["final_answer"])
    
    # Show data summary
    if result.get("fetched_data") is not None and not result["fetched_data"].empty:
        df = result["fetched_data"]
        print_divider()
        print("📊 DATA RETRIEVED")
        print_divider()
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head().to_string())
    
    # Note about visualization
    if result.get("figure") is not None:
        print_divider()
        print("📈 VISUALIZATION")
        print_divider()
        print("A chart was generated. Run with Streamlit to view: streamlit run app.py")


def interactive_mode():
    """Run in interactive REPL mode."""
    print_divider()
    print("🔮 Palantir OSDK Assistant - Interactive Mode")
    print_divider()
    print("Agent mode: Multi-Agent (Coordinator-Planner)")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for example queries")
    print_divider()
    
    conversation_history = []
    
    while True:
        try:
            query = input("\n🔹 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not query:
            continue
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break
        
        if query.lower() == "help":
            print("\nExample queries:")
            print("  • Show me all experiments")
            print("  • What surfactants are available?")
            print("  • Show results for experiment EXP001")
            print("  • Compare turbidity across different concentrations")
            print("  • List samples with Polysorbate 80")
            continue
        
        print("\n🤔 Thinking...\n")
        
        try:
            result = run_multi_agent_query(query, conversation_history)
            
            print_result(result)
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": result["final_answer"]})
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def single_query_mode(query: str):
    """Run a single query and exit."""
    print_divider()
    print("🔮 Palantir OSDK Assistant")
    print_divider()
    print("Mode: Multi-Agent (Coordinator-Planner)")
    print(f"Query: {query}")
    print_divider()
    
    print("\n🤔 Processing...\n")
    
    try:
        result = run_multi_agent_query(query)
        
        print_result(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Palantir OSDK Assistant - Query your Foundry data with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Interactive mode
    python main.py "Show me all experiments"    # Single query

For the web UI, run:
    streamlit run app.py
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to run (omit for interactive mode)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Force interactive mode even with a query"
    )
    
    args = parser.parse_args()
    
    if args.query and not args.interactive:
        single_query_mode(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
