# Azure OpenAI 版：第 2 环，智能体循环。

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# 让 Windows 终端能稳定打印中文。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 从工作区根目录读取 .env，沿用你前面 stage1 的配置方式。
# override=True 可以避免终端里旧的同名环境变量把 .env 的值盖掉。
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
    return {"error": f"Unknown tool: {name}"}


# 创建 Azure OpenAI 客户端。
# model 位置传的是 Azure 部署名，不是公开模型名。
client = AzureOpenAI(
    api_key=require_env("AZURE_OPENAI_API_KEY"),
    azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview"),
)

# 这里定义一个工具。Anthropic 文档里的 input_schema，
# 在 Azure OpenAI 里对应 function.parameters。
# 模型不会真的执行函数，它只会“提出想调用哪个工具、带什么参数”。
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
    }
]

# 第 2 环不再只发一轮，而是维护完整对话历史。
# 每轮都会把新消息追加到 messages 里，再继续请求模型。
messages = [
    {
        "role": "user",
        "content": "Schedule a weekly team standup every Monday at 9am for the next 4 weeks. Invite the whole team: alice@example.com, bob@example.com, carol@example.com.",
    }
]

response = client.chat.completions.create(
    model=require_env("AZURE_OPENAI_DEPLOYMENT"),
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0,
)

# 循环直到模型不再请求工具。
# Azure/OpenAI 风格里，判断条件不是 stop_reason，而是是否存在 tool_calls。
while True:
    choice = response.choices[0]
    message = choice.message
    tool_calls = message.tool_calls or []
    print(f"finish_reason: {choice.finish_reason}")

    if not tool_calls:
        break

    # 先把 assistant 这轮发出的 tool_calls 追加进历史。
    # 后续 role="tool" 的结果必须接在这条 assistant 消息后面。
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

    # Anthropic 第 2 环示例只处理一个 tool_use。
    # 这里也先按单工具思路写，但保留 for 循环，便于以后过渡到多工具。
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