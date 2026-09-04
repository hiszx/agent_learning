# Azure OpenAI 版：第 3 环，多个工具，并行调用。

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# 让 Windows 终端能稳定打印中文。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 从工作区根目录读取 .env，沿用前面示例的配置方式。
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def require_env(name: str) -> str:
    # 缺变量时直接报错，避免请求发出去后才发现配置不完整。
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"缺少环境变量: {name}")


def run_tool(name: str, arguments: dict) -> dict:
    # 这里代表“你的真实工具代码”。
    # 教学示例里不接真实日历系统，只返回一个假结果。
    if name == "create_calendar_event":
        return {
            "event_id": "evt_123",
            "status": "created",
            "title": arguments["title"],
        }
    if name == "list_calendar_events":
        return {
            "events": [
                {
                    "title": "Existing meeting",
                    "start": "14:00",
                    "end": "15:00",
                }
            ]
        }
    return {"error": f"Unknown tool: {name}"}
# 创建 Azure OpenAI 客户端。
# model 位置传的是 Azure 部署名，不是公开模型名。
client = AzureOpenAI(
    api_key=require_env("AZURE_OPENAI_API_KEY"),
    azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview"),
)

# Anthropic 的 tools[].input_schema，
# 在 Azure OpenAI 里对应 tools[].function.parameters。
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a calendar event with attendees and optional recurrence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recurrence": {
                        "type": "object",
                        "properties": {
                            "frequency": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly"],
                            },
                            "count": {"type": "integer", "minimum": 1},
                        },
                    },
                },
                "required": ["title", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List all calendar events on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD date"},
                },
                "required": ["date"],
            },
        },
    },
]

# 第 3 环继续维护完整对话历史，区别是这次模型可能在一轮里
# 返回多个 tool_calls，因此要把这一轮所有结果都补回去。
messages = [
    {
        "role": "user",
        "content": "Check what I have next Monday, then schedule a planning session that avoids any conflicts.",
    }
]

response = client.chat.completions.create(
    model=require_env("AZURE_OPENAI_DEPLOYMENT"),
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0,
)

while True:
    choice = response.choices[0]
    message = choice.message
    tool_calls = message.tool_calls or []
    print(f"finish_reason: {choice.finish_reason}")

    if not tool_calls:
        break

    # 一条 assistant 消息里可能带多个工具调用，
    # 后面的多个 role="tool" 结果都要接在它后面。
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

    # 第 3 环的重点不是外层 while，而是这一轮里可能有多个工具调用。
    # 因此这里要遍历当前 assistant 消息中的所有 tool_calls，
    # 并在再次请求模型前，把它们的结果全部补回历史。
    for tool_call in tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Input: {tool_call.function.arguments}")
        arguments = json.loads(tool_call.function.arguments)
        result = run_tool(tool_call.function.name, arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    response = client.chat.completions.create(
        model=require_env("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

final_message = response.choices[0].message
print(final_message.content or "")