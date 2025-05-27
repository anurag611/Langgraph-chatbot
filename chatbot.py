# chatbot.py

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import write_dot

# Load environment variables (optional)
from dotenv import load_dotenv

load_dotenv()


# Define state schema for messages
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the graph builder
graph_builder = StateGraph(State)

# Initialize your LLM (OpenAI GPT-4.1 in this case)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Ensure key is set
llm = init_chat_model("openai:gpt-4.1")

# Define the web search tool (Tavily)
tool = TavilySearch(max_results=2)
tools = [tool]

# Bind tools to the LLM so it can call them
llm_with_tools = llm.bind_tools(tools)


# Chatbot node uses LLM with tools
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# Tool node handles tool invocations
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Conditional edges for routing flow
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Compile the graph
graph = graph_builder.compile()

# ASCII Visualizer (Console)
if __name__ == "__main__":
    graph.get_graph().print_ascii()

    # Optional: export DOT file
    graph.get_graph().write_dot("chatbot_graph.dot")

nx_graph = graph.get_graph()  # This is a networkx DiGraph
write_dot(nx_graph, "chatbot_graph.dot")
