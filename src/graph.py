import os
import sys

# Add the src directory to the Python path so we can import from utils and agent modules
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledGraph
from langchain_core.runnables.graph import MermaidDrawMethod

from agent.state import ChatState
from agent.web_searcher import search_web
from agent.document_retriver import retrieve_documents
from agent.chatter import chatter_agent
from agent.query_rewriter import rewrite_query
from agent.route_question import query_router_agent
from agent.answer_generator import answer_generator
from agent.document_relevence_checker import check_relevance


def save_graph_as_png(app: CompiledGraph, output_file_path) -> None:
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    file_path = output_file_path if len(output_file_path) > 0 else "graph.png"
    with open(file_path, "wb") as f:
        f.write(png_image)
    return file_path


def build_rag_graph():
    # === Initialize Graph ===
    builder = StateGraph(ChatState)

    # === Add Agent Nodes ===
    builder.add_node("web_searcher", search_web)
    builder.add_node("document_retriever", retrieve_documents)
    builder.add_node("chatter", chatter_agent)
    # builder.add_node("query_rewriter", rewrite_query)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("relevance_grader", check_relevance)
    builder.add_node("query_router", query_router_agent)

    # === Start node ===
    builder.add_edge(START, "query_router")
    # builder.set_entry_point("query_router")
    # === Query Router decides routing ===
    builder.add_conditional_edges(
        "query_router",
        lambda state: state.metadata.get("signal", "chatter"),
        path_map={
            "Websearch": "web_searcher",
            "Vectorstore": "document_retriever",
            "Chatter": "chatter",
        },
    )

    # === Document retrieval and grading ===
    builder.add_edge("document_retriever", "relevance_grader")
    builder.add_conditional_edges(
        "relevance_grader",
        lambda state: state.metadata.get("relevance_score", "fail"),
        path_map={
            "fail": "web_searcher",
            "pass": "answer_generator",
        },
    )

    # === Web search path ===
    builder.add_edge("web_searcher", "answer_generator")

    # === Rerouting paths ===
    # builder.add_edge("query_rewriter", "document_retriever")
    # builder.add_edge("answer_generator", "evaluate_answer")

    # === End paths ===
    builder.add_edge("answer_generator", END)
    builder.add_edge("chatter", END)

    return builder.compile()


def test():
    agent = build_rag_graph()
    save_graph_as_png(agent, "test_graph.png")


if __name__ == "__main__":
    test()
