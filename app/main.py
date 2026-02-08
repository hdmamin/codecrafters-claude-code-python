import argparse
from functools import wraps
import json
import os
import sys
from typing import Callable

from openai import OpenAI


class ToolRegistry:

    available = {}

    @classmethod
    def tool(cls, func: Callable):
        cls[func.__name__] = func

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
def read(file_path: str):
    with open(file_path, "r") as f:
        return f.read()


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
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=TOOLS
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    for tool in chat.choices[0]["message"]["tool_calls"]:
        results = ToolRegistry.call_tool(
            tool["function"]["name"],
            json.loads(tool["arguments"])
        )
        print(results)

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
