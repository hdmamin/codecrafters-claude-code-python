import argparse
from functools import wraps
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Optional

from openai import OpenAI


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

SKILL_DIR = ".claude/skills"


def load_skills(skill_dir: str = SKILL_DIR) -> dict:
    """
    Returns
        dict mapping skill name (str) to dict with keys:
            frontmatter 
            description
            content
            file_path
            path
    """
    res = {}
    skill_dir = Path(skill_dir).expanduser()
    if not skill_dir.exists():
        return res

    for path in skill_dir.iterdir():
        if path.is_dir():
            file_path = path/"SKILL.md"
            with open(file_path, "r") as f:
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

                    res[frontmatter["name"]] = {
                        "description": frontmatter["description"],
                        "content": content.split("---", 2)[-1].strip(),
                        "file_path": str(file_path),
                        "path": str(file_path.parent)
                    }
                except Exception as e:
                    logger.warning(
                        f"frontmatter parsing failed for skill {path/'SKILL.md'}: {e}"
                    )    
    return res


def format_skill_frontmatters(skills: dict[str, dict]) -> str:
    return "\n".join(
        f"- {name}: {skill['description']}"
        for name, skill in skills.items()
    )


def is_skill_invocation(message: str, skills: dict) -> bool:
    """Return true if a user message invokes as skill (e.g. `/foo bar` where foo is a skill),
    False otherwise.
    """
    return message.startswith("/") and message[1:].split(" ")[0] in skills


def parse_skill_invocation(message: str, skills: dict) -> list[dict]:
    """Parse a user's skill invocation command like:

    /review foo bar

    into a dict like:

    {
        "skill": "review",
        "args": {
            "$ARGUMENTS": "foo bar",
            "$0": "foo",
            "$ARGUMENTS[0]": "foo",
            "$1": "bar",
            "$ARGUMENTS[1]": "bar",
        }
    }

    We return a list of these skill dicts in the order they are invoked.
    """
    words = message.split(" ")
    skill_names = []
    for i, word in enumerate(words):
        if word.startswith("/") and word[1:] in skills:
            skill_names.append(word[1:])
        else:
            args = words[i:]
            break
    else:
        args = []

    res = []
    for name in skill_names:
        cur = {
            "skill": name.lstrip("/"),
            "args": {
                "$ARGUMENTS": " ".join(args),
            }
        }
        for i, arg in enumerate(args):
            cur["args"][f"${i}"] = arg
            cur["args"][f"$ARGUMENTS[{i}]"] = arg
        res.append(cur)
    return res


def render_skill_messages(invocation: str, skills: dict) -> list[dict[str, str]]:
    """
    Args:
        invocation: str input by the user to invoke a skill, e.g. `/review foo bar`
        skills: dict returned by load_skills containing each skill's content, description, etc
    """
    path_info = (
        "Skill: {name} (located at {path})\nPaths in the instructions below are relative to that "
        "folder. Skill contents have been injected below.\n\n"
    )
    skill_infos = parse_skill_invocation(invocation, skills)
    templates = []
    for info in skill_infos:
        skill = skills[info["skill"]]
        template = skill["content"]
        for k, v in info["args"].items():
            template = template.replace(k, v)
        templates.append(
            {
                "role": "user",
                "content": path_info.format(name=info["skill"], path=skill["path"]) + template
            }
        )
    return templates


SKILLS = load_skills()
SKILL_FRONTMATTERS = format_skill_frontmatters(SKILLS)


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

        try:
            return tool(**arguments)
        except Exception as e:
            return f"FailedToolCall({str(e)})"


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
    res = subprocess.run(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, shell=True
    )
    return res.stderr or res.stdout


@ToolRegistry.tool
def Skill(name: str, args: Optional[list[str]] = None):
    """
    Args:
        name: Name of the skill to use.
        args: Optional args to hydrate the skill prompt.
    """
    skill = SKILLS[name]
    return skill["content"].format(*args)


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
                    "description": "The path to the file to read. You cannot run "
                                   "this on directories!"
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
    {
        "type": "function",
        "function": {
            "name": "Skill",
            "description": "Load a skill's instructions into the conversation",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                "name": { "type": "string", "description": "The name of the skill to use"},
                "args": { "type": "string", "description": "Optional arguments for the skill" }
                }
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

    if is_skill_invocation(args.p, SKILLS):
        user_content = render_skill_messages(args.p, SKILLS)
    else:
        user_content = [{"role": "user", "content": args.p}]

    messages = [
        {
            "role": "system",
            "content": f"You have access to the following skills:\n\n{SKILL_FRONTMATTERS}"
                       "If a skill matches the user's request, call the Skill tool with its name "
                       "and follow the instructions it returns."
        },
        *user_content
    ]
    # TODO: investigate why agent is apparently trying to read/write/run 'cd' as a file. Must be
    # gettingn confused by something about my path injection info, even though on first glance
    # it looks right to my eyes.
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
                # if tool.function.name.lower() == "read":
                #     print(messages)
                #     print(vars(tool)) # TODO rm
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
