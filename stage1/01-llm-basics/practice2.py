import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# 需要：pip install openai python-dotenv
# 在工作区根目录的 .env 文件中配置：
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com
#   AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
# 可选：
#   AZURE_OPENAI_API_VERSION=2025-02-01-preview

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"缺少环境变量: {name}")


client = AzureOpenAI(
    api_key=require_env("AZURE_OPENAI_API_KEY"),
    azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview"),
)

prompts = {
    "中文": "用一句话描述一只猫在做什么。",
    "English": "Describe in one sentence what a cat is doing.",
}

# for label, prompt in prompts.items():
#     prompt_tokens = []
#     completion_tokens = []

#     for _ in range(5):
#         response = client.chat.completions.create(
#             model=require_env("AZURE_OPENAI_DEPLOYMENT"),
#             max_completion_tokens=80,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         usage = response.usage
#         prompt_tokens.append(usage.prompt_tokens)
#         completion_tokens.append(usage.completion_tokens)
#     print(
#         f"[{label}] prompt_tokens={prompt_tokens[0]} "
#         f"completion min/max/mean={min(completion_tokens)}/{max(completion_tokens)}/{statistics.mean(completion_tokens):.1f}"
#     )


for label, prompt in prompts.items():
    prompt_tokens = []
    completion_tokens = []
    for _ in range(4):
        response = client.chat.completions.create(
            model = require_env("AZURE_OPENAI_DEPLOYMENT"),
            max_completion_tokens=80,
            messages = [{"role": "user", "content" : prompt}]
        )
        usage = response.usage
        prompt_tokens.append(usage.prompt_tokens)
        completion_tokens.append(usage.completion_tokens)
    print(
        f"{label}: prompt_tokens = {prompt_tokens[0]}"
        f" completion min/max/mean = {min(completion_tokens)}/{max(completion_tokens)}/{statistics.mean(completion_tokens):.1f}"
    )
