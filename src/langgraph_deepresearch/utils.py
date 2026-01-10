from datetime import datetime
from rich.panel import Panel
from rich.console import Console
from pathlib import Path
import json

console = Console()

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %-d, %Y")


def get_current_dir() -> Path:

    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()



def format_message_content(message):
    """Convert message content to displayable string"""
    parts = []
    tool_calls_processed = False
    
    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))
    
    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")
    
    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))



# def format_messages(messages):
#     """Format and display a list of messages with Rich formatting"""
#     for m in messages:
#         msg_type = m.__class__.__name__.replace('Message', '')
#         content = format_message_content(m)

#         if msg_type == 'Human':
#             console.print(Panel(content, title="🧑 Human", border_style="blue"))
#         elif msg_type == 'Ai':
#             console.print(Panel(content, title="🤖 Assistant", border_style="green"))
#         elif msg_type == 'Tool':
#             console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
#         else:
#             console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))

