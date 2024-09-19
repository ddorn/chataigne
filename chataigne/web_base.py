import asyncio
from enum import StrEnum
import json
from typing import Callable
import streamlit as st
from .horizontal_layout import st_horizontal

from .messages import (
    MessageHistory,
    TextMessage,
    ImageMessage,
    ToolRequestMessage,
    ToolOutputMessage,
)
from .llms import OpenAILLM
from .tool import Tool


class Actions(StrEnum):
    ALLOW_AND_RUN = "✅ Allow and Run"
    DENY = "❌ Deny"
    EDIT = "✏️ Edit"


class ChatBackend:
    def __init__(self, messages: MessageHistory):
        self.all_tools: list[Tool] = []
        self.messages = messages
        self.model = OpenAILLM("GPT 4o Mini", "gpt-4o-mini")

    def tool[T: Callable](self, tool: T) -> T:
        """Decorator to register a tool in the chat."""
        self.all_tools.append(Tool.from_function(tool))
        return tool

    def add_user_input(self, text: str):
        new_part = TextMessage(text=text, is_user=True)
        self.messages.append(new_part)

    async def generate_answer(self):
        new_parts = await self.model("Be straightforward.", self.messages, self.all_tools)
        self.messages.extend(new_parts)

    def actions_for(self, part_index: int) -> list[Actions | str]:
        part = self.messages[part_index]

        if isinstance(part, ToolRequestMessage) and self.needs_processing(part_index):
            return [Actions.ALLOW_AND_RUN, Actions.DENY]  # , Actions.EDIT]
        else:
            return []
            return [Actions.EDIT]

    def needs_processing(self, index: int) -> bool:
        part = self.messages[index]

        # There are two cases where we need to process a message:
        # 1. It's a tool request which has no corresponding tool output.
        # 2. It's a user message that has no response yet.
        if isinstance(part, ToolRequestMessage):
            return not any(
                isinstance(p, ToolOutputMessage) and p.id == part.id for p in self.messages
            )
        elif isinstance(part, TextMessage) and part.is_user:
            return index == len(self.messages) - 1
        else:
            return False

    def tool_requests_ids(self):
        return {m.id for m in self.messages if isinstance(m, ToolRequestMessage)}

    def tool_output_ids(self):
        return {m.id for m in self.messages if isinstance(m, ToolOutputMessage)}

    def tool_from_name(self, name: str) -> Tool:
        return next(t for t in self.all_tools if t.name == name)

    async def call_action(self, action: Actions, index: int):
        part = self.messages[index]

        if action == Actions.ALLOW_AND_RUN:
            assert isinstance(part, ToolRequestMessage)
            tool = self.tool_from_name(part.name)

            out = await tool.run(**part.parameters)
            self.messages.append(ToolOutputMessage(id=part.id, name=tool.name, content=out))

        elif action == Actions.DENY:
            assert isinstance(part, ToolRequestMessage)
            tool = self.tool_from_name(part.name)
            self.messages.append(
                ToolOutputMessage(
                    id=part.id, name=tool.name, content="Tool call denied by user", canceled=True
                )
            )

        # elif action == Actions.EDIT:

        else:
            raise NotImplementedError(action)

    def needs_generation(self) -> bool:
        if not self.messages:
            return False

        last = self.messages[-1]
        if isinstance(last, TextMessage):
            return last.is_user
        elif isinstance(last, ImageMessage):
            return True
        else:
            return False


class WebChat(ChatBackend):
    def __init__(self):
        messages = st.session_state.setdefault("messages", MessageHistory([]))

        super().__init__(messages)

        self.tool_requests_containers: dict = {}  # {part.id: st.container}

        # While developing, and the script reloads, the class of the messages
        # get out of sync, as the class is redefined/reimported. This is a
        # workaround to fix that.
        classes = [TextMessage, ImageMessage, ToolRequestMessage, ToolOutputMessage]
        for message in self.messages:
            message.__class__ = next(c for c in classes if c.__name__ == message.__class__.__name__)

    async def main(self):

        st.title("Diego's AI Chat")
        st.button("Clear chat", on_click=lambda: st.session_state.pop("messages"))

        with st.expander("Tools"):
            for tool in self.all_tools:
                st.write(f"### {tool.name}")
                st.write(tool.description)
                st.write(f"Parameters: {tool.parameters}")
                st.write(f"Required: {tool.required}")
                st.code(json.dumps(tool.to_openai(), indent=2), language="json")

        with st.expander("Message history as JSON"):
            st.code(json.dumps(self.messages.to_openai(), indent=2), language="json")

        st.markdown(
            """
            <style>
                [data-testid=stChatMessage] {
                    padding: 0.5rem;
                    margin: -0.5rem 0;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for i, message in enumerate(self.messages):

            # We put tool outputs next to its tool.
            if isinstance(message, ToolOutputMessage):
                with self.tool_requests_containers[message.id]:
                    st.write(f"➡ {message.content}")
                continue

            # Others have their own containers.
            role = message.to_openai()["role"]
            container = st.chat_message(name=role)
            with container:
                if isinstance(message, TextMessage):
                    st.write(message.text)
                elif isinstance(message, ImageMessage):
                    st.warning("Image messages are not supported yet.")
                elif isinstance(message, ToolRequestMessage):
                    self.tool_requests_containers[message.id] = container
                    s = f"Request to use **{message.name}**\n"
                    for key, value in message.parameters.items():
                        s += f"{key}: {self.to_inline_or_code_block(value)}\n"
                    st.write(s)
                else:
                    st.warning(f"Unsupported message type: {type(message)}")

                actions = self.actions_for(i)
                if actions:
                    with st_horizontal():
                        for action in actions:
                            if st.button(action, key=f"action_{action}_{i}"):
                                await self.call_action(action, i)
                                st.rerun()

        # If the last message is a user message, we need to ask for a new one.
        if self.needs_generation():
            await self.generate_answer()
            st.rerun()

        some_tool_was_not_run = any(
            isinstance(m, ToolRequestMessage) and m.id not in self.tool_output_ids()
            for m in self.messages
        )
        user_message = st.chat_input(disabled=some_tool_was_not_run)
        if user_message:
            self.add_user_input(user_message)
            st.rerun()

        return

    def to_inline_or_code_block(self, value):
        if "\n" in str(value):
            return f"\n```\n{value}\n```"
        else:
            return f"`{value}`"

    def run(self):
        asyncio.run(self.main())
