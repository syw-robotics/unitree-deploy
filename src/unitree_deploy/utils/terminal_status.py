from __future__ import annotations

from collections.abc import Iterable
import shutil
import sys


RESET = "\033[0m"
DIM = "\033[2m"
COLORS = {
    "white": "\033[37m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "bright_blue": "\033[94m",
    "bold red": "\033[1;31m",
}


def color(text: str, style: str) -> str:
    return f"{COLORS.get(style, '')}{text}{RESET}"


def visible_len(text: str) -> int:
    length = 0
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
        elif in_escape and char == "m":
            in_escape = False
        elif not in_escape:
            length += 1
    return length


class ComponentConsole:
    def __init__(self, name: str, style: str) -> None:
        self.name = name
        self.style = style
        self.status_active = False
        self.last_status_width = 0

    def log(self, message: str, *, style: str = "white") -> None:
        if self.status_active:
            sys.stdout.write("\n")
            self.status_active = False
        prefix = color(f"[{self.name}] ", self.style)
        sys.stdout.write(f"{prefix}{color(message, style)}\n")
        sys.stdout.flush()

    def status(self, fields: Iterable[tuple[str, str, str]]) -> None:
        parts = [color(f"[{self.name}]", self.style)]
        for label, value, style in fields:
            parts.append(f"{DIM}{label}{RESET}={color(value, style)}")
        line = "  ".join(parts)

        width = shutil.get_terminal_size((120, 20)).columns
        plain_width = visible_len(line)
        padding = max(self.last_status_width - plain_width, 0)
        self.last_status_width = min(max(self.last_status_width, plain_width), width)
        sys.stdout.write("\r" + line[: width + len(line) - plain_width] + (" " * padding))
        sys.stdout.flush()
        self.status_active = True

    def stop(self) -> None:
        if self.status_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.status_active = False
            self.last_status_width = 0
