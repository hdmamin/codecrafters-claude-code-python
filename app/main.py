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
    with open(file_path, "r") as f:
        return f.read()


def Write(file_path: str, content: str) -> None:
	with open(file_path, "w") as f:
		f.write(content)


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
	},
	}
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        messages.append(
            {
                "role": "assistant",
                "content": chat.choices[0].message.content,
                "tool_calls": chat.choices[0].message.tool_calls,
            }
        )

        tool_calls = chat.choices[0].message.tool_calls
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
