# main.py
from agent import agent, Context
from rich.console import Console
from memory_store import save_conversation_memory, clear_memory

console = Console()

def main():
    console.print("[bold green]🧠 Memory Chat Agent[/bold green]")
    console.print("[dim]Commands: /mem (view memory), /clear (clear memory), /exit (quit)[/dim]")
    user_id = input("Enter your user ID: ").strip() or "default_user"

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "/exit"]:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        if user_input == "/mem":
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "recall user info and get recent conversation"}]},
                context=Context(user_id=user_id)
            )
        elif user_input == "/clear":
            clear_memory()
            console.print("[bold red]Memory cleared![/bold red]")
            continue
        else:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                context=Context(user_id=user_id)
            )

        # Extract the response message
        if "messages" in response:
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                msg = last_message.content
            else:
                msg = str(last_message)
        else:
            msg = str(response)
        console.print(f"[bold cyan]🤖 Agent:[/bold cyan] {msg}")
        
        # Save conversation to long-term memory
        if user_input not in ["/mem", "/clear"]:
            save_conversation_memory(user_id, user_input, msg)


if __name__ == "__main__":
    main()
