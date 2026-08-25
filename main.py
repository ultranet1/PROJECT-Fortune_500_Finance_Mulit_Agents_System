"""
CLI entrypoint for the Fortune-500 finance multi-agent system.

Usage
-----
    python main.py                       # interactive chat, supervisor architecture
    python main.py --architecture swarm  # interactive chat, swarm architecture
    python main.py -q "Who is Apple's CEO?"

Requires a local Ollama server running with the target model pulled, e.g.:
    ollama pull llama3.1
    ollama serve
See .env.example for optional OLLAMA_HOST configuration.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from finance_agents.utils import pretty_print_messages

ARCHITECTURES = {
    "supervisor": "finance_agents.graphs.supervisor_agent",
    "swarm": "finance_agents.graphs.swarm_agent",
    "react": "finance_agents.graphs.react_agent",
    "conditional": "finance_agents.graphs.conditional_agent",
    "linear": "finance_agents.graphs.linear_agent",
}


def _build_graph(architecture: str, model_name: str):
    import importlib

    module = importlib.import_module(ARCHITECTURES[architecture])
    return module.build_graph(model_name=model_name)


def _needs_config(architecture: str) -> bool:
    # Only the checkpointed multi-agent graphs take a thread/user config.
    return architecture in {"supervisor", "swarm"}


def run(architecture: str, model_name: str, question: str | None) -> None:
    graph = _build_graph(architecture, model_name)
    config = {"configurable": {"thread_id": "cli", "user_id": "cli"}} if _needs_config(architecture) else None

    def ask(text: str) -> None:
        payload = {"messages": [{"role": "user", "content": text}]}
        stream = graph.stream(payload, config) if config else graph.stream(payload)
        for chunk in stream:
            pretty_print_messages(chunk)

    if question:
        ask(question)
        return

    print(f"Fortune-500 finance assistant ({architecture} architecture). Ctrl-C to exit.\n")
    try:
        while True:
            text = input("You: ").strip()
            if not text:
                continue
            ask(text)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fortune-500 finance multi-agent system")
    parser.add_argument(
        "--architecture",
        "-a",
        choices=list(ARCHITECTURES),
        default="supervisor",
        help="Which agent architecture to run (default: supervisor)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="llama3.1",
        help="Ollama model tag to use for every agent (default: llama3.1). Must be pulled locally.",
    )
    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="Ask a single question and exit instead of starting an interactive chat",
    )
    args = parser.parse_args()

    run(args.architecture, args.model, args.question)


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
