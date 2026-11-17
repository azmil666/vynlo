import json
from typing import Any


class FakeLLM:
    """Scriptable Fake LLM client returning scripted responses or raising scripted errors."""

    def __init__(
        self,
        responses: list[str | dict | Exception] | None = None,
        default_response: str | dict | None = None,
    ):
        self.responses: list[str | dict | Exception] = list(responses or [])
        if default_response is None:
            self.default_response = '{"clips": []}'
        elif isinstance(default_response, (dict, list)):
            self.default_response = json.dumps(default_response)
        else:
            self.default_response = str(default_response)
        self.calls: list[dict[str, str]] = []

    def add_response(self, response: str | dict | Exception) -> None:
        self.responses.append(response)

    def generate(self, system_instruction: str, user_content: str) -> str:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "user_content": user_content,
            }
        )

        if self.responses:
            next_resp = self.responses.pop(0)
            if isinstance(next_resp, Exception):
                raise next_resp
            if isinstance(next_resp, (dict, list)):
                return json.dumps(next_resp)
            return str(next_resp)

        return self.default_response

    def __call__(self, system_instruction: str, user_content: str) -> str:
        return self.generate(system_instruction, user_content)

    def reset(self) -> None:
        self.responses.clear()
        self.calls.clear()
