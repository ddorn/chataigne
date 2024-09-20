from enum import StrEnum
import json
from pathlib import Path
from typing import Callable

import streamlit as st
from streamlit_pills import pills as st_pills


from .horizontal_layout import st_horizontal
from .messages import (
    AnyMessagePart,
    MessageHistory,
    TextMessage,
    ImageMessage,
    ToolRequestMessage,
    ToolOutputMessage,
)
from .llms import EchoLLM, OpenAILLM, LLM, AnthropicLLM
from .tool import Tool

CSS_FILE = Path(__file__).parent / "styles.css"


class Actions(StrEnum):
    ALLOW_AND_RUN = "✅ Allow and Run"
    DENY = "❌ Deny"
    EDIT = "✏️ Edit"
    DELETE = "🗑️"


class ChatBackend:
    def __init__(self, messages: MessageHistory, model: LLM):
        self.tools: dict[str, Tool] = {}
        self.messages = messages
        self.model = model

    def tool[T: Callable](self, tool_function: T) -> T:
        """Decorator to register a tool in the chat."""
        tool = Tool.from_function(tool_function)
        if tool.name in self.tools:
            raise ValueError(f"A tool named {tool} is already registered.")
        else:
            self.tools[tool.name] = tool
        return tool_function

    def enabled_tools(self) -> list[Tool]:
        return [tool for tool in self.tools.values() if tool.enabled]

    def add_user_input(self, text: str):
        new_part = TextMessage(text=text, is_user=True)
        self.messages.append(new_part)

    def generate_answer(self) -> list[AnyMessagePart]:
        """Generates a new answer from the model and appends it to the messages."""
        new_parts = self.model("Be straightforward.", self.messages, self.enabled_tools())
        self.messages.extend(new_parts)
        return new_parts

    def actions_for(self, part_index: int) -> list[Actions | str]:
        part = self.messages[part_index]

        if isinstance(part, ToolRequestMessage) and self.needs_processing(part_index):
            return [Actions.ALLOW_AND_RUN, Actions.DENY, Actions.DELETE]  # , Actions.EDIT]
        else:
            return [Actions.DELETE]
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

    def call_action(self, action: Actions | str, index: int):
        part = self.messages[index]

        if action == Actions.ALLOW_AND_RUN:
            assert isinstance(part, ToolRequestMessage)
            tool = self.tools[part.name]

            out = tool.run(**part.parameters)
            self.messages.append(ToolOutputMessage(id=part.id, name=tool.name, content=out))

        elif action == Actions.DENY:
            assert isinstance(part, ToolRequestMessage)
            tool = self.tools[part.name]
            self.messages.append(
                ToolOutputMessage(
                    id=part.id, name=tool.name, content="Tool call denied by user", canceled=True
                )
            )

        elif action == Actions.DELETE:
            self.messages.pop(index)
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
    def __init__(self, models: list[LLM] = []):
        if not models:
            models = [
                OpenAILLM("GPT 4o", "gpt-4o"),
                OpenAILLM("GPT 4o Mini", "gpt-4o-mini"),
                AnthropicLLM("Claude 3.5 Sonnet", "claude-3-5-sonnet-20240620"),
                EchoLLM(),
            ]
        self.available_models = models

        messages = st.session_state.setdefault("messages", MessageHistory([]))
        messages = MessageHistory(messages.model_dump())
        st.session_state.messages = messages
        super().__init__(messages, models[0])

        self.tool_requests_containers: dict = {}  # {part.id: st.container}

        # While developing, and the script reloads, the class of the messages
        # get out of sync, as the class is redefined/reimported. This is a
        # workaround to fix that.
        classes = [TextMessage, ImageMessage, ToolRequestMessage, ToolOutputMessage]
        for message in self.messages:
            message.__class__ = next(c for c in classes if c.__name__ == message.__class__.__name__)

    def show_sidebar(self):
        with st.sidebar:
            st.header("Activated tools")
            for name, tool in sorted(self.tools.items()):
                tool.enabled = st.toggle(name, True)

            st.header("Options")
            st.button("Clear chat", on_click=lambda: st.session_state.pop("messages"))

            if st.toggle("Show tools"):
                for tool in self.tools.values():
                    st.write(f"### {tool.name}")
                    st.write(tool.description)
                    st.code(json.dumps(tool.to_openai(), indent=2), language="json")

            if st.button("Show messages history"):

                @st.dialog("Messages history", width="large")
                def show_history():
                    kind = st_pills("Kind", ["Raw", "For OpenAI", "For Anthropic"])

                    if kind == "Raw":
                        st.write(self.messages.model_dump())
                    elif kind == "For OpenAI":
                        st.write(self.messages.to_openai())
                    elif kind == "For Anthropic":
                        st.write(self.messages.to_anthropic())
                    else:
                        raise ValueError(kind)

                show_history()

    def main(self):
        self.inject_css()
        st.title("Chataigne 🌰")

        if len(self.available_models) > 1:
            models_by_name = {model.nice_name: model for model in self.available_models}
            model_name = st_pills(
                "Model selection",
                sorted(models_by_name.keys()),
                label_visibility="collapsed",
                index=0,
                key="model_pill",
            )
            # model_name = st.radio(
            #     "Model selection", sorted(models_by_name.keys()), index=0, key="model_radio",
            #     horizontal=True)
            self.model = models_by_name[model_name]

        self.show_sidebar()

        for i in range(len(self.messages)):
            self.show_message(i)

        some_tool_was_not_run = any(
            isinstance(m, ToolRequestMessage) and m.id not in self.tool_output_ids()
            for m in self.messages
        )

        st.chat_input(
            disabled=some_tool_was_not_run,
            key="chat_input",
            on_submit=lambda: self.add_user_input(st.session_state.chat_input),
        )

        if self.needs_generation():
            new = self.generate_answer()
            for i in range(len(self.messages) - len(new), len(self.messages)):
                self.show_message(i)

    def show_message(self, index: int):
        message = self.messages[index]
        # We put tool outputs next to its tool.
        if isinstance(message, ToolOutputMessage):
            with self.tool_requests_containers[message.id]:
                st.write(f"➡ {message.content}")
                self.show_actions_for(index)
            return

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

            self.show_actions_for(index)

    def show_actions_for(self, index: int):
        actions = self.actions_for(index)
        if actions:
            with st_horizontal():
                for action in actions:
                    st.button(
                        action,
                        key=f"action_{action}_{index}",
                        on_click=self.call_action,
                        args=(action, index),
                    )

    def inject_css(self):
        st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)

    def to_inline_or_code_block(self, value):
        if "\n" in str(value):
            return f"\n```\n{value}\n```"
        else:
            return f"`{value}`"

    def show_in_modal(self, **kwargs):

        @st.dialog("Debug", width="large")
        def _():
            for key, value in kwargs.items():
                st.write(f"### {key}")
                st.write(value)

    def run(self):
        self.main()
