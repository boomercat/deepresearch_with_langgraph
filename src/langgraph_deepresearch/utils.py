# import Path
from datetime import datetime
from rich.panel import Panel
from rich.console import Console

console = Console()

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %-d, %Y")


# def get_current_dir() -> Path:

#     try:
#         return Path(__file__).resolve().parent
#     except NameError:  # __file__ is not defined
#         return Path.cwd()

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

