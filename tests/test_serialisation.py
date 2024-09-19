import pytest
from chataigne.messages import (
    TextMessage,
    ImageMessage,
    ToolRequestMessage,
    MessagePart,
    MessageHistory,
)


@pytest.mark.parametrize(
    "message",
    [
        TextMessage(text="User message 1", is_user=True),
        TextMessage(text="Assistant message 1", is_user=False),
        ImageMessage(base_64="image"),
        ToolRequestMessage(name="tool_1", parameters={"param1": "value1"}, id="1"),
        # ToolOutputMessage(name="tool_1", parameters={"param1": "value1"}, id="1"),
    ],
)
def test_message_serialise_and_back(message: MessagePart):
    as_dict = message.model_dump()
    back = MessageHistory.model_validate([as_dict])[0]

    assert back == message


# Test loading a list of messages
def test_message_history_serialise_and_back():
    messages = MessageHistory(
        [
            TextMessage(text="User message 1", is_user=True),
            TextMessage(text="Assistant message 1", is_user=False),
            ImageMessage(base_64="image=="),
            ToolRequestMessage(
                name="tool_1",
                parameters={"param1": 17.5, "param2": "value2"},
                id="1",
            ),
        ]
    )

    expected = [
        {"type": "text", "text": "User message 1", "is_user": True},
        {"type": "text", "text": "Assistant message 1", "is_user": False},
        {"type": "image", "base_64": "image=="},
        {
            "type": "toolrequest",
            "name": "tool_1",
            "parameters": {"param1": 17.5, "param2": "value2"},
            "id": "1",
        },
    ]

    result = messages.model_dump()

    assert result == expected
