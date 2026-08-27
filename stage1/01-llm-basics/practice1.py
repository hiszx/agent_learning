import os
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI

# 需要：pip install openai python-dotenv
# 在当前目录的 .env 文件中配置：
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com
#   AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
# 可选：
#   AZURE_OPENAI_API_VERSION=2025-02-01-preview

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


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

response = client.chat.completions.create(
    model=require_env("AZURE_OPENAI_DEPLOYMENT"),
    max_completion_tokens=100,
    messages=[{"role": "user", "content": "用一句话自我介绍。"}],
)

text = response.choices[0].message.content or ""
print("回应：", text)
print("usage:", response.usage)

finish_reason = response.choices[0].finish_reason
print("finish_reason:", finish_reason)
assert finish_reason in ("stop", "length"), f"非预期 finish_reason: {finish_reason}"
assert len(text) > 0, "回应不应为空"
assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0, "token 数应 > 0"
print("✅ 练习 1 通过 — 你已成功打通 Azure OpenAI API")