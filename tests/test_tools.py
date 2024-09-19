from chataigne.tool import create_model_from_function


def test_create_model_from_function():
    def custom_add(a: int, b: float = 2):
        """Add two numbers"""
        return a + b * 2

    model = create_model_from_function(custom_add)
    instance = model(a=1, b=4.0)
    assert instance.model_dump() == {"a": 1, "b": 4.0}
    shema = model.model_json_schema()
    assert shema["required"] == ["a"]
    assert list(shema["properties"]) == ["a", "b"]
