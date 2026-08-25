"""
Single agent, conditional routing.

START -> llm --(tool call?)--> tools -> llm --(tool call?)--> END

Improves on ``linear_agent`` by letting the LLM decide, via LangGraph's
built-in ``tools_condition``, whether a tool call is actually required.
Inputs that don't need a tool (e.g. "My name is Alaran") now fall through
directly to a normal chatbot response instead of always visiting the tool
node.
"""

from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from finance_agents.tools import ALL_TOOLS


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model_name: str = "llama3.1"):
    graph_builder = StateGraph(State)

    llm = ChatOllama(model=model_name)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def llm_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools=ALL_TOOLS)

    graph_builder.add_node("llm", llm_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "llm")
    graph_builder.add_conditional_edges(
        "llm",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph_builder.add_edge("tools", "llm")

    return graph_builder.compile()


if __name__ == "__main__":
    from finance_agents.utils import pretty_print_messages

    graph = build_graph()
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}
    ):
        pretty_print_messages(chunk)
