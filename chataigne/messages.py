import base64
import itertools
import json
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Annotated, Any, Literal

import PIL.Image
from anthropic.types import MessageParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field, RootModel

__all__ = [
    "TextMessage",
    "ImageMessage",
    "ToolRequestMessage",
    "ToolOutputMessage",
    "MessagePart",
]


class MessagePart(BaseModel, ABC):
    type: Any

    @abstractmethod
    def to_openai(self):
        raise NotImplementedError()

    @abstractmethod
    def to_anthropic(self):
        raise NotImplementedError()

    # @classmethod
    # def all_subclasses(cls):
    #     return set(cls.__subclasses__()).union(
    #         [s for c in cls.__subclasses__() for s in c.all_subclasses()]
    #     )


class TextMessage(MessagePart):
    text: str
    is_user: bool
    type: Literal["text"] = "text"

    def to_openai(self):
        return {
            "role": "user" if self.is_user else "assistant",
            "content": [{"type": "text", "text": self.text}],
        }

    to_anthropic = to_openai


class ImageMessage(MessagePart):
    base_64: str
    type: Literal["image"] = "image"

    @classmethod
    def from_path(cls, path: str):
        img = PIL.Image.open(path)
        # Convert the image to PNG format
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # Encode the image to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return cls(base_64=img_base64)

    def to_openai(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + self.base_64,
                    },
                }
            ],
        }

    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self.base_64,
                    },
                }
            ],
        }


class ToolRequestMessage(MessagePart):
    name: str
    parameters: dict[str, Any]
    id: str
    type: Literal["toolrequest"] = "toolrequest"

    def to_openai(self):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": self.id,
                    "type": "function",
                    "function": {"name": self.name, "arguments": json.dumps(self.parameters)},
                }
            ],
        }

    def to_anthropic(self):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": self.name,
                    "input": self.parameters,
                    "id": self.id,
                }
            ],
        }


class ToolOutputMessage(MessagePart):
    id: str
    name: str
    content: str
    canceled: bool = False
    type: Literal["tooloutput"] = "tooloutput"

    def to_openai(self):
        return {
            "role": "tool",
            # "name": self.name,
            "content": self.content,
            "tool_call_id": self.id,
        }

    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.id,
                    "content": self.content,
                }
            ],
        }


type AnyMessagePart = TextMessage | ImageMessage | ToolRequestMessage | ToolOutputMessage


class MessageHistory(RootModel[list[AnyMessagePart]]):
    root: list[Annotated[AnyMessagePart, Field(discriminator="type")]]

    def __init__(self, root: list[AnyMessagePart]):
        super().__init__(root=root)

    def __iter__(self):  # type: ignore
        return iter(self.root)

    def __getitem__(self, index):
        return self.root[index]

    def __len__(self):
        return len(self.root)

    def __add__(self, other):
        return MessageHistory(self.root + other.root)

    def append(self, other: AnyMessagePart):
        self.root.append(other)

    def extend(self, other: list[AnyMessagePart]):
        self.root.extend(other)

    def remove(self, other: AnyMessagePart):
        self.root.remove(other)

    def pop(self, index: int):
        return self.root.pop(index)

    def index(self, value: AnyMessagePart):
        return self.root.index(value)

    def insert(self, index: int, value: AnyMessagePart):
        self.root.insert(index, value)

    def to_openai(self) -> list[ChatCompletionMessageParam]:
        formated = []
        # For openai, we need to merge:
        # - an optional assistant TextMessage and the consecutive ToolRequestMessages into a single one
        # - a user TextMessage and subsequent ImageMessages from the same user into a single one

        i = 0
        while i < len(self):
            message = self[i]

            # Merge consecutive user text message and Image messages
            if isinstance(message, TextMessage) and message.is_user:
                new = message.to_openai()
                i += 1
                for attached_image in itertools.takewhile(
                    lambda x: isinstance(x, ImageMessage), self[i:]
                ):
                    new = merge(new, attached_image.to_openai())
                    i += 1

            # Merge an assistant message with subsequent tool requests
            elif isinstance(message, TextMessage) and not message.is_user:
                new = message.to_openai()
                i += 1
                for tool_request in itertools.takewhile(
                    lambda x: isinstance(x, ToolRequestMessage), self[i:]
                ):
                    new = merge(new, tool_request.to_openai())
                    i += 1

            # Just add the message
            else:
                new = message.to_openai()
                i += 1

            formated.append(new)

        return formated

    def to_anthropic(self) -> list[MessageParam]:
        formated = []

        # For anthropic, we need to merge:
        # - all user messages (text and image) and tool outputs into a single message
        # - all other, ie: all assistant messages and tool requests into a single message

        def is_user_message(x):
            return (
                (isinstance(x, TextMessage) and x.is_user)
                or isinstance(x, ImageMessage)
                or isinstance(x, ToolOutputMessage)
            )

        i = 0
        while i < len(self):
            message = self[i]

            new = message.to_anthropic()
            # Merge consecutive user text/image message and tool outputs
            if is_user_message(message):
                i += 1
                for part in itertools.takewhile(is_user_message, self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1
            else:
                i += 1
                for part in itertools.takewhile(lambda x: not is_user_message(x), self[i:]):
                    new = merge(new, part.to_anthropic())
                    i += 1

            formated.append(new)

        return formated


def merge[T: (dict, list)](a: T, b: T) -> T:
    """
    Merge two dictionaries or lists together:
    - If both are lists, concatenate them
    - If both are dictionaries, merge them recursively. If a key is present in both dictionaries, the value must be the same.
    """
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, dict) and isinstance(b, dict):
        new = {}
        for key in a.keys():
            new[key] = a[key]
        for key in b.keys():
            if key in new and isinstance(new[key], (dict, list)):
                new[key] = merge(new[key], b[key])
            elif key in new:
                assert (
                    new[key] == b[key]
                ), f"Conflict on key {key}: {new[key]} != {b[key]}.\n{a}\n{b}"
            else:
                new[key] = b[key]
        return new
    else:
        raise ValueError(f"Cannot merge {a} and {b}")
