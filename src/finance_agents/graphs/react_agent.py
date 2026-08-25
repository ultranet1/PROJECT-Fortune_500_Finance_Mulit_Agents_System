"""
Single agent, built with LangGraph's ``create_react_agent`` prebuilt.

Functionally equivalent to ``conditional_agent``, but expressed in one call
instead of hand-wiring nodes and edges. This is the pattern the multi-agent
graphs (``swarm_agent``, ``supervisor_agent``) build on, since re-deriving
the low-level graph for every worker agent doesn't scale.
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from finance_agents.tools import ALL_TOOLS

PROMPT = """
You are an assistant for research and analysis of Fortune 500 companies. You have access to three tools:
- A Wikipedia tool for retrieving factual summary information about companies
- A stock performance data tool for retrieving stock price information from local CSV files
- A Python tool for executing Python code, which is to be used for creating stock performance visualizations
"""


def build_graph(model_name: str = "llama3.1"):
    llm = ChatOllama(model=model_name)
    return create_react_agent(llm, ALL_TOOLS, prompt=PROMPT)


if __name__ == "__main__":
    from finance_agents.utils import pretty_print_messages

    agent = build_graph()
    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Tell me Tesla's current CEO, their latest stock price, and generate a plot "
                        "of the closing price with the most up-to-date data you have available."
                    ),
                }
            ]
        }
    ):
        pretty_print_messages(chunk)
