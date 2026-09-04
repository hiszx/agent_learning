# Azure OpenAI 版：第 4 环，错误处理。

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
        if "attendees" in arguments and len(arguments["attendees"]) > 10:
            raise ValueError("Too many attendees (max 10)")
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
    raise ValueError(f"Unknown tool: {name}")


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

# 构造一个会触发工具异常的请求，观察模型如何读取错误结果并继续回复。
messages = [
    {
        "role": "user",
        "content": "Check my calendar on 2026-09-08, then schedule an all-hands titled Team All-Hands "
        "from 2026-09-08T10:00:00 to 2026-09-08T11:00:00 with everyone: "
        + ", ".join(f"user{i}@example.com" for i in range(17)),
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

        try:
            result = run_tool(tool_call.function.name, arguments)
            tool_content = json.dumps(
                {"ok": True, "result": result},
                ensure_ascii=False,
            )
        except Exception as exc:
            # 这里不让程序崩掉，而是把错误结果发回模型。
            # 当前这套 Azure Chat Completions 结构里没有 Anthropic 那种 is_error 字段，
            # 所以把失败状态和错误文本编码进 tool content 里。
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

    response = client.chat.completions.create(
        model=require_env("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

final_message = response.choices[0].message
print(final_message.content or "")