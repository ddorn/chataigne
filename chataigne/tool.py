from inspect import Parameter, signature
from typing import Any, Callable

from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import BaseModel, create_model


class Tool(BaseModel):
    name: str
    description: str
    pydantic_model: type[BaseModel]
    enabled: bool = True

    def shema(self) -> dict[str, Any]:
        schema = self.pydantic_model.model_json_schema()
        parameters = (schema["properties"],)
        required = (schema["required"],)

        return {
            "type": "object",
            "properties": parameters,
            "required": required,
        }

    def to_openai(self) -> ChatCompletionToolParam:
        # def as_json(self, for_anthropic: bool = False) -> dict:
        data = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.shema(),
            },
            "strict": True,
        }
        return ChatCompletionToolParam(**data)

    def to_anthropic(self) -> ToolParam:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.shema(),
        }

    def run(self, **kwargs) -> str:
        raise NotImplementedError()

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        FArgs = create_model_from_function(func)
        assert (
            func.__doc__ is not None
        ), f"Function '{func.__name__}' must have a docstring explaining how to use it (for the LLM)"

        class CustomTool(cls):
            def run(self, **kwargs):
                return str(func(**kwargs))

        return CustomTool(
            name=func.__name__,
            description=func.__doc__,
            pydantic_model=FArgs,
        )


def create_model_from_function(func) -> type[BaseModel]:
    # Get the signature of the function
    sig = signature(func)

    # Prepare the attributes for the Pydantic model
    attributes = {}
    for name, param in sig.parameters.items():
        if param.annotation is Parameter.empty:
            raise ValueError(f"Parameter '{name}' of function '{func.__name__}' has no type hint")
        param_type = param.annotation

        # Check if the parameter has a default value
        if param.default is Parameter.empty:
            attributes[name] = (param_type, ...)
        else:
            attributes[name] = (param_type, param.default)

    # Create and return the dynamic model
    name = func.__name__.capitalize() + "Args"
    return create_model(name, **attributes)
