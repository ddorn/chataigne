import anthropic
import pytest

from chataigne.messages import (
    ImageMessage,
    MessageHistory,
    TextMessage,
    ToolOutputMessage,
    ToolRequestMessage,
)

# Simple white 4x4 image, in base64 from PNG
image = "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAMAQMAAACHjHWnAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURScoIv///0T6qSMAAAABYktHRAH/Ai3eAAAAB3RJTUUH6AkSEBcOP97EywAAAAtJREFUCNdjYCAMAAAkAAEuHnGgAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDI0LTA5LTE4VDE2OjIzOjE0KzAwOjAwzJ8kxwAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyNC0wOS0xOFQxNjoyMzoxNCswMDowML3CnHsAAAAodEVYdGRhdGU6dGltZXN0YW1wADIwMjQtMDktMThUMTY6MjM6MTQrMDA6MDDq172kAAAAAElFTkSuQmCC"

# Define message and expected result pairs
empty_list_messages = ([], [])
single_user_text_message = (
    [TextMessage(text="Hello", is_user=True)],
    [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
)
single_assistant_text_message = (
    [TextMessage(text="Hello", is_user=False)],
    [{"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}],
)
user_assistant_user_text_messages = (
    [
        TextMessage(text="Hello", is_user=True),
        TextMessage(text="Processing", is_user=False),
        TextMessage(text="Goodbye", is_user=True),
    ],
    [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Processing"}]},
        {"role": "user", "content": [{"type": "text", "text": "Goodbye"}]},
    ],
)
user_text_and_image_message = (
    [TextMessage(text="Hello", is_user=True), ImageMessage(base_64=image)],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        }
    ],
)
assistant_text_and_tool_request_message = (
    [
        TextMessage(text="Processing", is_user=False),
        ToolRequestMessage(name="tool_name", parameters={"param": "value"}, id="1"),
    ],
    [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Processing"}],
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "tool_name", "arguments": '{"param": "value"}'},
                }
            ],
        }
    ],
)
mixed_messages = (
    [
        TextMessage(text="Hello", is_user=True),
        ImageMessage(base_64=image),
        TextMessage(text="Processing", is_user=False),
        ToolRequestMessage(name="tool_name", parameters={"param": "value"}, id="1"),
        ToolOutputMessage(id="1", name="tool_name", content="tool output"),
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Processing"}],
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "tool_name", "arguments": '{"param": "value"}'},
                }
            ],
        },
        {
            "role": "tool",
            "content": "tool output",
            "tool_call_id": "1",
        },
    ],
)
multiple_user_and_assistant_messages_with_tools = (
    [
        TextMessage(text="User message 1", is_user=True),
        TextMessage(text="Assistant message 1", is_user=False),
        ToolRequestMessage(name="tool_1", parameters={"param1": "value1"}, id="1"),
        TextMessage(text="User message 2", is_user=True),
        TextMessage(text="Assistant message 2", is_user=False),
        ToolRequestMessage(name="tool_2", parameters={"param2": "value2"}, id="2"),
    ],
    [
        {"role": "user", "content": [{"type": "text", "text": "User message 1"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant message 1"}],
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "tool_1", "arguments": '{"param1": "value1"}'},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "User message 2"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant message 2"}],
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "tool_2", "arguments": '{"param2": "value2"}'},
                }
            ],
        },
    ],
)
interleaved_messages_with_different_types = (
    [
        TextMessage(text="User text 1", is_user=True),
        ImageMessage(base_64=image),
        TextMessage(text="Assistant text 1", is_user=False),
        ToolRequestMessage(name="tool_1", parameters={"param1": "value1"}, id="1"),
        TextMessage(text="User text 2", is_user=True),
        TextMessage(text="Assistant text 2", is_user=False),
        ToolRequestMessage(name="tool_2", parameters={"param2": "value2"}, id="2"),
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "User text 1"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant text 1"}],
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "tool_1", "arguments": '{"param1": "value1"}'},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "User text 2"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant text 2"}],
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {"name": "tool_2", "arguments": '{"param2": "value2"}'},
                }
            ],
        },
    ],
)
tool_request_before_assistant_message = (
    [
        ToolRequestMessage(name="tool_1", parameters={"param1": "value1"}, id="1"),
        TextMessage(text="Assistant text 1", is_user=False),
    ],
    [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Assistant text 1"}],
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "tool_1", "arguments": '{"param1": "value1"}'},
                }
            ],
        }
    ],
)


@pytest.mark.parametrize(
    "messages,expected_result",
    [
        empty_list_messages,
        single_user_text_message,
        single_assistant_text_message,
        user_assistant_user_text_messages,
        # user_text_and_image_message,
        # assistant_text_and_tool_request_message,
        # mixed_messages,
        # multiple_user_and_assistant_messages_with_tools,
        # interleaved_messages_with_different_types,
    ],
)
def test_to_anthropic(messages, expected_result):
    result = MessageHistory(messages).to_anthropic()
    assert result == expected_result


@pytest.mark.xfail
@pytest.mark.parametrize(
    "messages,expected_result",
    [
        # Merging them is not implemented in this order, but it should like be done.
        # Though we never build lists that have output messages before tool requests.
        tool_request_before_assistant_message,
    ],
)
def test_to_anthropic_fails(messages, expected_result):
    result = MessageHistory(messages).to_anthropic()
    assert result == expected_result


@pytest.mark.skip
@pytest.mark.parametrize(
    "messages",
    [
        single_user_text_message,
        single_assistant_text_message,
        user_assistant_user_text_messages,
        user_text_and_image_message,
        assistant_text_and_tool_request_message,
        mixed_messages,
        multiple_user_and_assistant_messages_with_tools,
        interleaved_messages_with_different_types,
        tool_request_before_assistant_message,
    ],
)
def test_anthropic_accepts_messages(messages):
    messages = messages[1]

    anthropic.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1,
    )
