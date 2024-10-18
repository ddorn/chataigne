import ast

from chataigne.web_base import WebChat
from chataigne.tools.amazing_marvin import add_marvin_task


app = WebChat()

app.tool(add_marvin_task)


@app.tool
def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


@app.tool
def example(x: int, y: float, z: bool, w: list[int], a: dict, b: str | None = None):
    """An example tool"""
    return "\n".join(f"{k}: {type(v)}" for k, v in locals().items())


@app.tool
def calc(expression: str):
    """Computes a mathematical expression using python ast.literal_eval"""

    return ast.literal_eval(expression)


if __name__ == "__main__":
    app.run()
