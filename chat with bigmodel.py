from pydantic import BaseModel, Field
import requests
import json
from typing import Iterator


DEBUG = True


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="GLM4/",
            description="Prefix to be added before model names.",
        )
        GLM4_API_BASE_URL: str = Field(
            default="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            description="Base URL for accessing GLM-4 API endpoints.",
        )
        GLM4_API_KEY: str = Field(
            default="",
            description="API key for authenticating requests to the GLM-4 API.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.models = [
            "glm-4-plus",
            "glm-4-air",
            "glm-4-air-0111",
            "glm-4-airx",
            "glm-4-airx",
            "glm-4-long",
            "glm-4-flashx",
            "glm-4-flash",
            "glm-zero-preview",
        ]

    def pipes(self):
        """
        Open WebUI expects a pipes() method that returns available models.
        """
        return [
            {"id": f"{self.valves.NAME_PREFIX}{model}", "name": f"GLM-4 {model}"}
            for model in self.models
        ]

    def pipe(self, body: dict, __user__: dict):
        """
        Handles requests to GLM-4 models, supporting streaming.
        """
        if DEBUG:
            print(f"Received body: {json.dumps(body, indent=2)}")

        # 提取模型 ID（去掉前缀）
        model_id = body["model"].split("/")[-1]
        if model_id not in self.models:
            return f"Error: Unknown GLM-4 model '{model_id}'. Available models: {', '.join(self.models)}."

        headers = {
            "Authorization": f"Bearer {self.valves.GLM4_API_KEY}",
            "Content-Type": "application/json",
        }

        # 组装请求体（保留 open webui 传递的所有参数）
        payload = {**body, "model": model_id}

        if DEBUG:
            print(f"Sending request to GLM-4 API: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(
                self.valves.GLM4_API_BASE_URL,
                json=payload,
                headers=headers,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            if body.get("stream", False):
                return self.handle_streaming(response)
            else:
                return response.json()

        except Exception as e:
            if DEBUG:
                print(f"GLM-4 Pipe Error: {e}")
            return f"Error: {e}"

    def handle_streaming(self, response) -> Iterator[str]:
        """Handles streaming responses from the GLM-4 API."""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data:"):
                    content = decoded_line[5:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        data = json.loads(content)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        if DEBUG:
                            print(f"Failed to parse JSON: {content}")
