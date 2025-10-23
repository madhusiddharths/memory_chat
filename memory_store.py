# memory_store.py
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import DuckDB
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Force simple storage for now to avoid API issues
print("Using simple file-based storage to avoid API issues")
embeddings = None

# Persistent DuckDB vector store
PERSIST_DIR = os.path.join(os.getcwd(), "duckdb_store")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize storage system
if embeddings is not None:
    try:
        # Use DuckDB vector store with embeddings
        if os.path.exists(os.path.join(PERSIST_DIR, "memory.duckdb")):
            store = DuckDB(persist_directory=PERSIST_DIR, embedding=embeddings)
        else:
            store = DuckDB.from_texts([], embedding=embeddings, persist_directory=PERSIST_DIR)
        print("✅ Using DuckDB vector store with embeddings")
    except Exception as e:
        print(f"Warning: Failed to initialize DuckDB store: {e}")
        store = None
else:
    store = None
    print("✅ Using simple file-based storage")

# Simple file-based storage functions
MEMORY_FILE = os.path.join(PERSIST_DIR, "simple_memory.json")
import json

def load_simple_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_simple_memory(memories):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memories, f, indent=2)

# 🧠 Utility functions
def save_memory(namespace: str, text: str, metadata: Dict[str, Any]):
    """Store text and metadata persistently."""
    if store is not None:
        try:
            store.add_texts(texts=[text], metadatas=[metadata], namespace=namespace)
        except Exception as e:
            print(f"Warning: Failed to save memory: {e}")
    else:
        # Use simple file-based storage
        try:
            memories = load_simple_memory()
            memory_entry = {
                "namespace": namespace,
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            memories.append(memory_entry)
            save_simple_memory(memories)
            print(f"✅ Saved to simple memory: {text[:50]}...")
        except Exception as e:
            print(f"Warning: Failed to save to simple memory: {e}")

def search_memory(namespace: str, query: str, k: int = 3):
    """Retrieve similar memories based on query text."""
    if store is not None:
        try:
            results = store.similarity_search(query=query, namespace=namespace, k=k)
            return [
                {"text": r.page_content, "metadata": r.metadata}
                for r in results
            ]
        except Exception as e:
            print(f"Warning: Failed to search memory: {e}")
            return []
    else:
        # Simple text-based search
        try:
            memories = load_simple_memory()
            matching_memories = []
            query_lower = query.lower()
            
            for memory in memories:
                if memory.get("namespace") == namespace:
                    text_lower = memory.get("text", "").lower()
                    if query_lower in text_lower or any(word in text_lower for word in query_lower.split()):
                        matching_memories.append({
                            "text": memory["text"],
                            "metadata": memory["metadata"]
                        })
            
            return matching_memories[:k]
        except Exception as e:
            print(f"Warning: Failed to search simple memory: {e}")
            return []

def get_memory(namespace: str, user_id: str):
    """Get specific memory for a user."""
    if store is not None:
        try:
            results = store.similarity_search(query=user_id, namespace=namespace, k=1)
            if results:
                return {"text": results[0].page_content, "metadata": results[0].metadata}
            return None
        except Exception:
            return None
    else:
        # Simple file-based search
        try:
            memories = load_simple_memory()
            for memory in memories:
                if memory.get("namespace") == namespace and user_id in memory.get("text", ""):
                    return {"text": memory["text"], "metadata": memory["metadata"]}
            return None
        except Exception:
            return None

def save_conversation_memory(user_id: str, user_message: str, agent_response: str):
    """Save conversation turn to long-term memory."""
    timestamp = datetime.now().isoformat()
    conversation_text = f"User: {user_message}\nAgent: {agent_response}"
    metadata = {
        "user_id": user_id,
        "type": "conversation",
        "timestamp": timestamp,
        "user_message": user_message,
        "agent_response": agent_response
    }
    save_memory("conversations", conversation_text, metadata)

def retrieve_relevant_memories(user_id: str, query: str, k: int = 5):
    """Retrieve relevant memories based on semantic similarity."""
    # Search in conversations
    conversation_memories = search_memory("conversations", query, k)
    # Search in user info
    user_memories = search_memory("users", query, k)
    
    # Combine and deduplicate
    all_memories = conversation_memories + user_memories
    return all_memories[:k]

def get_conversation_history(user_id: str, limit: int = 10):
    """Get recent conversation history for context."""
    try:
        results = store.similarity_search(
            query=f"user_id:{user_id}",
            namespace="conversations", 
            k=limit
        )
        return [
            {"text": r.page_content, "metadata": r.metadata}
            for r in results
        ]
    except Exception:
        return []

def clear_memory():
    """Reset all stored memory."""
    for f in os.listdir(PERSIST_DIR):
        os.remove(os.path.join(PERSIST_DIR, f))
    print("🧹 Memory cleared.")
