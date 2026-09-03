# Azure OpenAI 版：第 1 环，单个工具，单轮对话。

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# 让 Windows 终端能稳定打印中文。
if hasattr(sys.stdout, "reconfigure"):
	sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 从工作区根目录读取 .env。
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
			"start": arguments["start"],
			"end": arguments["end"],
			"attendees": arguments.get("attendees", []),
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

# 第 1 环里只保留一条 user message，方便看清最小闭环。
user_message = {
	"role": "user",
	"content": "Schedule a 30-minute sync with alice@example.com and bob@example.com on Monday, March 30, 2026 at 10am.",
}

# 第一次请求：把用户消息和工具定义一起发给模型。
response = client.chat.completions.create(
	model=require_env("AZURE_OPENAI_DEPLOYMENT"),
	messages=[user_message],
	tools=tools,
	tool_choice="auto",
	temperature=0,
)

choice = response.choices[0]
message = choice.message
tool_call = (message.tool_calls or [None])[0]

print(f"finish_reason: {choice.finish_reason}")
if tool_call is None:
	raise RuntimeError("模型没有请求工具，无法继续第 1 环示例")

print(f"Tool: {tool_call.function.name}")
print(f"Input: {tool_call.function.arguments}")

# tool_call.function.arguments 是 JSON 字符串，要先反序列化。
arguments = json.loads(tool_call.function.arguments)
result = run_tool(tool_call.function.name, arguments)

# 第二次请求：把模型上一条 tool call 和工具结果一起发回去。
followup = client.chat.completions.create(
	model=require_env("AZURE_OPENAI_DEPLOYMENT"),
	messages=[
		user_message,
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
			],
		},
		{
			"role": "tool",
			"tool_call_id": tool_call.id,
			"content": json.dumps(result, ensure_ascii=False),
		},
	],
	tools=tools,
	tool_choice="auto",
	temperature=0,
)

final_choice = followup.choices[0]
final_message = final_choice.message
print(f"finish_reason: {final_choice.finish_reason}")
print(final_message.content or "")
