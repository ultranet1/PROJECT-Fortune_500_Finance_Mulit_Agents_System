"""
Two agents, arranged as a swarm.

A ``researcher`` (Wikipedia + stock data tools) and an ``analyst`` (Python
REPL tool) can hand off control to one another directly, using
``langgraph_swarm.create_handoff_tool``. There is no central coordinator;
whichever agent is active decides whether to keep working or transfer to
its peer.

Requires ``langgraph-swarm`` (see requirements.txt).
"""

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

from finance_agents.tools import python_repl_tool, stock_data_tool, wikipedia_tool


def build_graph(model_name: str = "llama3.1"):
    llm = ChatOllama(model=model_name)

    # Handoff tool: analyst -> researcher
    transfer_to_researcher = create_handoff_tool(
        agent_name="researcher",
        description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or load stock performance data from CSV files.",
    )

    # Handoff tool: researcher -> analyst
    transfer_to_analyst = create_handoff_tool(
        agent_name="analyst",
        description="Transfer user to the analyst assistant, who can create visualizations of provided data.",
    )

    research_agent = create_react_agent(
        llm,
        tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
        prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
        name="researcher",
    )

    analyst_agent = create_react_agent(
        llm,
        [python_repl_tool, transfer_to_researcher],
        prompt="You generate plots of stock performance data provided by another assistant.",
        name="analyst",
    )

    checkpointer = InMemorySaver()

    swarm = create_swarm(
        agents=[research_agent, analyst_agent],
        default_active_agent="researcher",
    ).compile(checkpointer=checkpointer)

    return swarm


if __name__ == "__main__":
    from finance_agents.utils import pretty_print_messages

    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    swarm = build_graph()
    for chunk in swarm.stream(
        {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]}, config
    ):
        pretty_print_messages(chunk)
