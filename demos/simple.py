from chataigne.web_base import WebChat
from chataigne.tools.amazing_marvin import add_marvin_task


app = WebChat()

app.tool(add_marvin_task)


@app.tool
def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


if __name__ == "__main__":
    app.run()
