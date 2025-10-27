import os
from colorama import Fore, Style
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from agent.state import ChatState
from utils.logging import get_caller_logger
from utils.cleaner import clean_documents

from dotenv import load_dotenv

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

logger = get_caller_logger()

# Limit how many results to preview in logging
max_preview = 3

web_search_tool = TavilySearch(max_results=5, days=365, search_depth="advanced", include_answer=True, tavily_api_key=tavily_api_key)  # Uses advanced search depth for more accurate results  # Include a short answer to original query in the search results.  # You have defined this API key in the .env file.


def search_web(state: ChatState) -> ChatState:
    """WebSearcher

    Args:
        state (ChatState): current conversation state

    Returns:
        ChatState: new conversation state
    """
    logger.info("\n---WEB SEARCH---")
    # Start time

    question = state.question
    web_results = web_search_tool.invoke(question)

    # Tavily (or other search tools) may return a dict containing a 'results' list
    results_list = []
    if isinstance(web_results, dict) and "results" in web_results and isinstance(web_results["results"], list):
        results_list = web_results["results"]
    elif isinstance(web_results, list):
        results_list = web_results
    else:
        # Single dict with fields that describe the answer; wrap it
        results_list = [web_results]

    documents = []
    for doc in results_list:
        if isinstance(doc, dict):
            # Try common fields for textual content
            content = doc.get("content") or doc.get("answer") or doc.get("summary") or doc.get("snippet") or doc.get("text") or ""
            # If the top-level doc contains nested 'results', flatten them
            if not content and "results" in doc and isinstance(doc["results"], list):
                for inner in doc["results"]:
                    inner_content = None
                    if isinstance(inner, dict):
                        inner_content = inner.get("content") or inner.get("answer") or inner.get("summary") or inner.get("snippet") or inner.get("text")
                    if inner_content:
                        documents.append(Document(metadata=dict(url=inner.get("url", ""), title=inner.get("title", "")), page_content=inner_content))
                continue

            documents.append(Document(metadata=dict(url=doc.get("url", ""), title=doc.get("title", "")), page_content=content))
        elif isinstance(doc, str):
            documents.append(Document(metadata={}, page_content=doc))
        else:
            documents.append(Document(metadata={}, page_content=str(doc)))
    documents, stats = clean_documents(documents, verbose=True)

    state.documents.extend(documents)

    for i, doc in enumerate(documents[:max_preview]):
        # print(type(doc))
        logger.info(f"{Fore.MAGENTA}🔹 Result {i+1}{Style.RESET_ALL}\n" f"   {Fore.GREEN}Title:{Style.RESET_ALL} {doc.metadata.get('title', 'N/A')}\n" f"   {Fore.BLUE}URL:{Style.RESET_ALL} {doc.metadata.get('url', 'N/A')}\n" f"   {Fore.YELLOW}Excerpt:{Style.RESET_ALL} {doc.page_content[:200].strip()}...\n")

    if len(documents) > max_preview:
        logger.info(f"{Fore.CYAN}...and {len(documents) - max_preview} more results not shown.{Style.RESET_ALL}")

    logger.info(f"Total number of web search documents: {len(documents)}")
    return state
