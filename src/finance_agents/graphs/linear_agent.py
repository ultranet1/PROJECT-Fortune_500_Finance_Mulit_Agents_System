"""
Single agent, linear routing.

START -> llm -> tools -> END

Every LLM turn is unconditionally routed through the tool node, whether a
tool call was produced or not. This is the simplest possible tool-calling
graph and is mainly useful for understanding LangGraph's node/edge syntax;
see ``conditional_agent`` for a version that only calls tools when needed.
"""

from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
    graph_builder.add_edge("llm", "tools")
    graph_builder.add_edge("tools", END)

    return graph_builder.compile()


if __name__ == "__main__":
    from finance_agents.utils import pretty_print_messages

    graph = build_graph()
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}
    ):
        pretty_print_messages(chunk)
