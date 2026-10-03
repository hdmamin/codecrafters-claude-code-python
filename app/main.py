import argparse
from functools import wraps
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from openai import OpenAI


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class ToolRegistry:

    available = {}

    @classmethod
    def tool(cls, func: Callable):
        cls.available[func.__name__] = func

        @wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    @classmethod
    def call_tool(cls, name: str, arguments: dict):
        try:
            tool = cls.available[name]
        except KeyError as e:
            raise ValueError(f"tool {name!r} not found.")
        return tool(**arguments)


@ToolRegistry.tool
def Read(file_path: str) -> str:
    """Read text from a file."""
    with open(file_path, "r") as f:
        return f.read()


@ToolRegistry.tool
def Write(file_path: str, content: str) -> None:
    """Write text to a file."""
    with open(file_path, "w") as f:
        f.write(content)
    return content


@ToolRegistry.tool
def Bash(command: str) -> str:
    """Run a bash command."""
    res = subprocess.run(command.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return res.stderr or res.stdout


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                    "type": "string",
                    "description": "The path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                }
            }
        }
    },
]


def load_skill_info(skill_dir: str = ".claude/skills", as_str: bool = True) -> str | list[dict]:
    """
    Returns
        list of frontmatter dicts containingn keys "name" and "description" if as_str=False.
        If as_str=True, we stringify this into a bulleted list.
    """
    frontmatters = []
    skill_dir = Path(skill_dir).expanduser()
    if not skill_dir.exists():
        return "" if as_str else []

    for path in skill_dir.iterdir():
        if path.is_dir():
            with open(path/"SKILL.md", "r") as f:
                content = f.read()
                try:
                    frontmatter = re.findall(r"(?<=---\n).*(?=\n---\n)", content, re.DOTALL)
                    if len(frontmatter) != 1:
                        raise ValueError(
                            f"Expected 1 frontmatter, found {len(frontmatter)}: {frontmatter}"
                        )
                    frontmatter = dict(line.split(": ") for line in frontmatter[0].splitlines())
                    if any(key not in frontmatter for key in ("name", "description")):
                        raise ValueError("Frontmatter did not contain all expected keys.")
                except Exception as e:
                    logger.warning(
                        f"frontmatter parsing failed for skill {path/'SKILL.md'}: {e}"
                    )    
                else:
                    frontmatters.append(frontmatter)
    if as_str:
        return "\n".join(f"- {f['name']}: {f['description']}" for f in frontmatters)
    return frontmatters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    skill_info = load_skill_info(as_str=True)
    messages = [
        {"role": "system", "content": f"You have access to the following skills:\n\n{skill_info}"},
        {"role": "user", "content": args.p},
    ]
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        tool_calls = chat.choices[0].message.tool_calls
        # Need to see what messages array looks like as it grows, find expected vals and compare
        messages.append(
            {
                "role": "assistant",
                "content": chat.choices[0].message.content,
                "tool_calls": chat.choices[0].message.tool_calls,
            }
        )

        if tool_calls:
            for tool in tool_calls:
                results = ToolRegistry.call_tool(
                    tool.function.name,
                    json.loads(tool.function.arguments)
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": results,
                        "tool_call_id": tool.id,
                    }
                )
        else:
            break
    print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
