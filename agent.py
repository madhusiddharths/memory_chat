# agent.py
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from memory_store import (
    store, save_memory, get_memory, save_conversation_memory, 
    retrieve_relevant_memories, get_conversation_history
)


# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

# Create the model instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


@dataclass
class Context:
    user_id: str


class UserInfo(TypedDict):
    name: str
    language: str


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info into memory."""
    user_id = runtime.context.user_id
    namespace = "users"
    text = f"User {user_id}: {user_info['name']} speaks {user_info['language']}"
    metadata = {"user_id": user_id, "type": "user_info"}
    save_memory(namespace, text, metadata)
    return f"Saved info for {user_info['name']}."


@tool
def recall_user_info(runtime: ToolRuntime[Context]) -> str:
    """Recall stored user info."""
    user_id = runtime.context.user_id
    namespace = "users"
    info = get_memory(namespace, user_id)
    return f"Known user info: {info}" if info else "No memory found."

@tool
def search_memories(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search for relevant memories based on a query."""
    user_id = runtime.context.user_id
    memories = retrieve_relevant_memories(user_id, query, k=3)
    if memories:
        memory_texts = [mem["text"] for mem in memories]
        return f"Found relevant memories: {'; '.join(memory_texts)}"
    return "No relevant memories found."

@tool
def get_recent_conversation(runtime: ToolRuntime[Context]) -> str:
    """Get recent conversation history for context."""
    user_id = runtime.context.user_id
    history = get_conversation_history(user_id, limit=5)
    if history:
        history_texts = [h["text"] for h in history]
        return f"Recent conversation: {'; '.join(history_texts)}"
    return "No recent conversation history found."


agent = create_agent(
    model=llm,
    tools=[save_user_info, recall_user_info, search_memories, get_recent_conversation],
    store=store,
    context_schema=Context,
)
