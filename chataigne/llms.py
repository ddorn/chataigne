import json
import anthropic
import openai
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
)

from .tool import Tool
from .messages import MessageHistory, MessagePart, TextMessage, ToolRequestMessage


class LLM:
    def __init__(
        self,
        nice_name: str,
        model_name: str,
    ):
        self.nice_name = nice_name
        self.model_name = model_name

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:
        raise NotImplementedError()


class OpenAILLM(LLM):
    def __init__(self, nice_name: str, model_name: str):
        super().__init__(nice_name, model_name)
        self.client = openai.AsyncOpenAI()

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:
        answer = (
            (
                await self.client.chat.completions.create(
                    messages=[
                        ChatCompletionSystemMessageParam(content=system, role="system"),
                        *messages.to_openai(),
                    ],  # type: ignore
                    model=self.model_name,
                    temperature=0.2,
                    tools=[tool.to_openai() for tool in tools],
                )
            )
            .choices[0]
            .message
        )

        new_messages = []
        if answer.content is not None:
            if isinstance(answer.content, str):
                new_messages.append(TextMessage(text=answer.content, is_user=False))
            else:
                new_messages.append(
                    TextMessage(text=f"Unrecognized content type: {answer.content}", is_user=False)
                )

        if answer.tool_calls:
            for tool_call in answer.tool_calls:
                new_messages.append(
                    ToolRequestMessage(
                        name=tool_call.function.name,
                        parameters=json.loads(tool_call.function.arguments),
                        id=tool_call.id,
                    )
                )

        return new_messages


class AnthropicLLM(LLM):
    def __init__(self, nice_name: str, model_name: str):
        super().__init__(nice_name, model_name)
        self.client = anthropic.AsyncAnthropic()

    async def __call__(
        self, system: str, messages: MessageHistory, tools: list[Tool]
    ) -> list[MessagePart]:

        answer = await self.client.messages.create(
            system=system,
            messages=messages.to_anthropic(),
            model=self.model_name,
            temperature=0.2,
            max_tokens=4096,
            tools=[tool.to_anthropic() for tool in tools],
        )

        new_messages = []
        for part in answer.content:
            if part.type == "text":
                new_messages.append(TextMessage(text=part.text, is_user=False))
            elif part.type == "tool_use":
                assert isinstance(part.input, dict)
                new_messages.append(
                    ToolRequestMessage(name=part.name, parameters=part.input, id=part.id)
                )
            else:
                new_messages.append(
                    TextMessage(text=f"Unrecognized content type: {part.type}", is_user=False)
                )

        return new_messages


MODELS = [
    AnthropicLLM("Claude 3.5 Sonnet", "claude-3-5-sonnet-20240620"),
    OpenAILLM("GPT 4o", "gpt-4o"),
    OpenAILLM("GPT 4o mini", "gpt-4o-mini"),
]
