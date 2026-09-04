# Azure OpenAI 版：第 5 环，Tool Runner 抽象层。

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, get_args, get_origin

from dotenv import load_dotenv
from openai import AzureOpenAI

# 让 Windows 终端能稳定打印中文。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 从工作区根目录读取 .env，沿用前面示例的配置方式。
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"缺少环境变量: {name}")


def annotation_to_schema(annotation: Any) -> dict:
    # 这里只覆盖当前示例需要的几种类型，够用即可。
    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict:
        return {"type": "object"}

    if origin in (list, tuple):
        item_annotation = args[0] if args else str
        return {
            "type": "array",
            "items": annotation_to_schema(item_annotation),
        }

    # 兼容 PEP 604 语法，如 list[str] | None、dict | None。
    if args and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return annotation_to_schema(non_none_args[0])

    return {"type": "string"}


def parse_arg_descriptions(docstring: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    lines = docstring.splitlines()
    in_args_section = False

    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args_section = True
            continue
        if not in_args_section:
            continue
        if not stripped:
            continue
        if not line.startswith("    "):
            break
        if ":" not in stripped:
            continue
        name, description = stripped.split(":", 1)
        descriptions[name.strip()] = description.strip()

    return descriptions


class RunnableTool:
    def __init__(self, func, description: str, input_schema: dict):
        self.func = func
        self.name = func.__name__
        self.description = description
        self.input_schema = input_schema

    @classmethod
    def from_function(cls, func):
        docstring = inspect.getdoc(func) or ""
        description = docstring.splitlines()[0] if docstring else func.__name__
        arg_descriptions = parse_arg_descriptions(docstring)
        signature = inspect.signature(func)

        properties = {}
        required = []
        for parameter in signature.parameters.values():
            schema = annotation_to_schema(parameter.annotation)
            if parameter.name in arg_descriptions:
                schema["description"] = arg_descriptions[parameter.name]
            properties[parameter.name] = schema
            if parameter.default is inspect.Signature.empty:
                required.append(parameter.name)

        return cls(
            func=func,
            description=description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def invoke(self, arguments: dict) -> str:
        result = self.func(**arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


def azure_tool(func):
    return RunnableTool.from_function(func)


class ToolRunner:
    # Azure OpenAI 的 Chat Completions 没有现成的 Tool Runner SDK。
    # 这里手动封一层，把第 2~4 环的样板循环收起来。
    def __init__(self, client: AzureOpenAI, model: str, tools: list[RunnableTool], max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.tool_map = {tool.name: tool for tool in tools}

    def run(self, messages: list[dict]):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[tool.to_openai_tool() for tool in self.tools],
            tool_choice="auto",
            temperature=0,
        )

        while True:
            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []
            print(f"finish_reason: {choice.finish_reason}")

            if not tool_calls:
                return message

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Input: {tool_call.function.arguments}")
                arguments = json.loads(tool_call.function.arguments)
                tool = self.tool_map[tool_call.function.name]

                try:
                    tool_content = json.dumps(
                        {"ok": True, "result": json.loads(tool.invoke(arguments))},
                        ensure_ascii=False,
                    )
                except Exception as exc:
                    tool_content = json.dumps(
                        {"ok": False, "error": str(exc)},
                        ensure_ascii=False,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[tool.to_openai_tool() for tool in self.tools],
                tool_choice="auto",
                temperature=0,
            )


client = AzureOpenAI(
    api_key=require_env("AZURE_OPENAI_API_KEY"),
    azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview"),
)


@azure_tool
def create_calendar_event(
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    recurrence: dict | None = None,
) -> str:
    """Create a calendar event with attendees and optional recurrence.

    Args:
        title: Event title.
        start: Start time in ISO 8601 format.
        end: End time in ISO 8601 format.
        attendees: Email addresses to invite.
        recurrence: Dict with 'frequency' (daily, weekly, monthly) and 'count'.
    """
    if attendees and len(attendees) > 10:
        raise ValueError("Too many attendees (max 10)")
    return json.dumps({"event_id": "evt_123", "status": "created", "title": title})


@azure_tool
def list_calendar_events(date: str) -> str:
    """List all calendar events on a given date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    return json.dumps({"events": [{"title": "Existing meeting", "start": "14:00", "end": "15:00"}]})


runner = ToolRunner(
    client=client,
    model=require_env("AZURE_OPENAI_DEPLOYMENT"),
    tools=[create_calendar_event, list_calendar_events],
)

final_message = runner.run(
    messages=[
        {
            "role": "user",
            "content": "Check what I have next Monday, then schedule a planning session that avoids any conflicts.",
        }
    ]
)

print(final_message.content or "")