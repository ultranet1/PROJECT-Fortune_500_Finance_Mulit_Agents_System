"""
Two agents, coordinated by a supervisor.

A ``supervisor`` agent delegates work to a ``researcher`` (Wikipedia +
stock data tools) and an ``analyst`` (Python REPL tool), and is
responsible for composing their outputs into a single response back to
the user. Unlike the swarm architecture, the worker agents never talk to
each other directly - every handoff goes through the supervisor.

This is the architecture ``main.py`` runs by default, since it produces
the most coherent, single-response experience for an end user.

Requires ``langgraph-supervisor`` (see requirements.txt).
"""

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from finance_agents.tools import python_repl_tool, stock_data_tool, wikipedia_tool

SUPERVISOR_PROMPT = (
    "You are a supervisor managing two agents:\n"
    "- a research agent. Assign research and data collection tasks to this agent\n"
    "- an analyst agent. Assign the creation of visualizations via code to this agent\n"
    "Assign work to one agent at a time, do not call agents in parallel.\n"
    "Do not do any work yourself."
)

RESEARCH_PROMPT = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks, including looking-up factual information and stock data. DO NOT write any code.\n"
    "- After you're done with your tasks, respond to the supervisor directly\n"
    "- Respond ONLY with the results of your work, do NOT include ANY other text."
)

ANALYST_PROMPT = (
    "You are an agent that can run arbitrary Python code.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with tasks that require running code to produce an output.\n"
    "- After you're done with your tasks, respond to the supervisor directly\n"
    "- Respond ONLY with the results of your work, do NOT include ANY other text."
)


def build_graph(model_name: str = "llama3.1"):
    llm = ChatOllama(model=model_name)

    research_agent = create_react_agent(
        llm,
        tools=[wikipedia_tool, stock_data_tool],
        prompt=RESEARCH_PROMPT,
        name="researcher",
    )

    analyst_agent = create_react_agent(
        llm,
        [python_repl_tool],
        prompt=ANALYST_PROMPT,
        name="analyst",
    )

    checkpointer = InMemorySaver()

    supervisor = create_supervisor(
        model=llm,
        agents=[research_agent, analyst_agent],
        prompt=SUPERVISOR_PROMPT,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer)

    return supervisor


if __name__ == "__main__":
    from finance_agents.utils import pretty_print_messages

    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    supervisor = build_graph()
    for chunk in supervisor.stream(
        {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]}, config
    ):
        pretty_print_messages(chunk)
