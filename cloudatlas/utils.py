"""Utility module"""
import os
import warnings

# Turn off keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rich import print
import telegram_send


class RemoteMonitor:
    """Class for remote logging using telegram."""

    def __init__(self):
        try:
            import telegram_send

            print(f"Remote monitor [green]available[/green]")
        except ImportError:
            print(f"Remote monitor [red]unavailable[/red]")

        # Check if telegram_send is configured
        # ????

    def send(self, message):
        """Sends a message."""
        # Check wether it's a message or more than one
        if hasattr(message, "__iter__"):
            msg = message
        else:
            msg = [message]
        try:
            telegram_send.send(messages=msg)
        except Exception as e:
            print("An exception was raised while sending a message:")
            print(e)


